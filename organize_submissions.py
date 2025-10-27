import argparse
import hashlib
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
from zipfile import ZipFile, BadZipFile


ASSIGNMENT_PATTERN = re.compile(r"^(.+)_(.+)\.zip$")
NOISE_NAMES = {"__MACOSX", ".DS_Store"}
CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass
class CopyStats:
    copied: int = 0
    skipped_identical: int = 0


@dataclass
class FinalizeResult:
    zips_deleted: int = 0
    zip_delete_failures: int = 0
    students_moved: int = 0
    student_move_failures: int = 0


def resolve_assignment_name(zip_path: Path) -> str:
    """Resolve the assignment name from the Moodle zip file name."""
    _, assignment = _parse_zip_components(zip_path)
    return assignment


def resolve_course_prefix(zip_path: Path) -> str:
    """Resolve the course prefix from the Moodle zip file name."""
    course_prefix, _ = _parse_zip_components(zip_path)
    return course_prefix


def _parse_zip_components(zip_path: Path) -> tuple[str, str]:
    """Return (course_prefix, assignment_name) parsed from the zip filename."""
    match = ASSIGNMENT_PATTERN.match(zip_path.name)
    if not match:
        raise ValueError(f"Zip filename does not match expected pattern: {zip_path.name}")

    course_prefix = match.group(1).strip()
    assignment = match.group(2).strip()
    if not course_prefix:
        raise ValueError(f"Course prefix resolved to empty string for {zip_path.name}")
    if not assignment:
        raise ValueError(f"Assignment name resolved to empty string for {zip_path.name}")
    return course_prefix, assignment


def iter_student_dirs(extracted_root: Path) -> Iterator[Path]:
    """Yield directories that correspond to student submissions, flattening wrapper folders."""
    root = _collapse_singleton_dir(extracted_root)
    if root.name == "submissions" and root.parent != root:
        root = root.parent
    stack = [root]
    visited: set[Path] = set()

    while stack:
        current = stack.pop()
        try:
            resolved_current = current.resolve()
        except OSError:
            continue
        if resolved_current in visited:
            continue
        visited.add(resolved_current)

        submissions_dir = current / "submissions"
        if submissions_dir.is_dir():
            yield current
            continue

        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue

        for entry in entries:
            if entry.name in NOISE_NAMES or entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue

            stack.append(entry)


def normalize_student_name(
    raw_student_dir_name: str,
    assignment_name: Optional[str] = None,
    course_prefix: Optional[str] = None,
    zip_stem: Optional[str] = None,
) -> str:
    """
    Normalize the student directory name by stripping known course or assignment prefixes.
    Returns the canonical "<Name> - <ID>" string on success.
    """
    name = raw_student_dir_name.strip()
    if not name:
        raise ValueError("Student directory name is empty")

    prefixes = _gather_prefix_candidates(assignment_name, course_prefix, zip_stem)
    if prefixes:
        name = _strip_prefixes_from_left(name, prefixes)

    if " - " not in name:
        raise ValueError(f"Student directory name does not follow '<Name> - <ID>' pattern: {raw_student_dir_name}")

    left, right = name.rsplit(" - ", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Student directory name missing name or ID component: {raw_student_dir_name}")
    return f"{left} - {right}"


def copy_submissions(src_submissions_dir: Path, dest_assignment_dir: Path) -> CopyStats:
    """Copy first-level files from submissions into the destination assignment folder."""
    stats = CopyStats()
    if not src_submissions_dir.is_dir():
        return stats

    for entry in sorted(src_submissions_dir.iterdir()):
        if not entry.is_file():
            continue

        existing_path = dest_assignment_dir / entry.name
        if existing_path.exists() and _files_identical(entry, existing_path):
            logger.info("    Skipping identical file: %s", existing_path.name)
            stats.skipped_identical += 1
            continue

        dest_path = safe_copy(entry, dest_assignment_dir)
        if dest_path.name != entry.name:
            logger.info("    Copied %s as %s", entry.name, dest_path.name)
        else:
            logger.info("    Copied %s", entry.name)
        stats.copied += 1

    return stats


def safe_copy(src_file: Path, dest_dir: Path) -> Path:
    """Copy a file into dest_dir, adding numeric suffixes when name conflicts occur."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / src_file.name
    if not candidate.exists():
        shutil.copy2(src_file, candidate)
        return candidate

    base_name, suffix = _split_filename(candidate.name)
    counter = 2
    while True:
        candidate = dest_dir / f"{base_name} ({counter}){suffix}"
        if not candidate.exists():
            shutil.copy2(src_file, candidate)
            return candidate
        if _files_identical(src_file, candidate):
            return candidate
        counter += 1


def main(base_dir: Path) -> int:
    """Coordinate the submission organisation workflow."""
    if not base_dir.exists():
        logger.error("Base directory does not exist: %s", base_dir)
        return 1
    if not base_dir.is_dir():
        logger.error("Base path is not a directory: %s", base_dir)
        return 1

    work_dir = base_dir / ".work_extract"
    students_root = base_dir / "Students"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    encountered_errors = False
    zips_processed = 0
    students_touched = set()
    files_copied = 0
    files_skipped = 0

    try:
        zip_paths = sorted(base_dir.glob("*.zip"))
        if not zip_paths:
            logger.warning("No zip files found in %s", base_dir)

        for zip_path in zip_paths:
            try:
                course_prefix, assignment_name = _parse_zip_components(zip_path)
            except ValueError as exc:
                logger.error("Skipping %s: %s", zip_path.name, exc)
                encountered_errors = True
                continue

            logger.info("Processing %s → %s", zip_path.name, assignment_name)
            zip_stem = zip_path.stem.strip()
            extract_dest = work_dir / zip_path.stem
            if extract_dest.exists():
                shutil.rmtree(extract_dest)
            extract_dest.mkdir(parents=True, exist_ok=True)

            try:
                with ZipFile(zip_path) as zf:
                    _safe_extract(zf, extract_dest)
            except (BadZipFile, OSError, ValueError) as exc:
                logger.error("Failed to extract %s: %s", zip_path.name, exc)
                encountered_errors = True
                continue

            student_dirs = list(iter_student_dirs(extract_dest))
            if not student_dirs:
                logger.warning("No student directories found in %s", zip_path.name)
                continue

            for student_dir in student_dirs:
                try:
                    student_name = normalize_student_name(
                        student_dir.name,
                        assignment_name=assignment_name,
                        course_prefix=course_prefix,
                        zip_stem=zip_stem,
                    )
                except ValueError as exc:
                    logger.warning("  Skipping folder %s: %s", student_dir.name, exc)
                    encountered_errors = True
                    continue

                submissions_dir = student_dir / "submissions"
                if not submissions_dir.exists() or not submissions_dir.is_dir():
                    logger.info("  No submissions for %s", student_name)
                    students_touched.add(student_name)
                    continue

                dest_student_dir = students_root / student_name
                dest_assignment_dir = dest_student_dir / assignment_name
                stats = copy_submissions(submissions_dir, dest_assignment_dir)
                students_touched.add(student_name)

                if stats.copied == 0 and stats.skipped_identical == 0:
                    _cleanup_empty_assignment_dir(dest_assignment_dir, students_root)
                    logger.info("  No files copied for %s", student_name)
                else:
                    logger.info(
                        "  %s: %d file(s) copied, %d duplicate(s) skipped",
                        student_name,
                        stats.copied,
                        stats.skipped_identical,
                    )

                files_copied += stats.copied
                files_skipped += stats.skipped_identical

            zips_processed += 1
    finally:
        if encountered_errors:
            logger.warning("Temporary extraction directory retained at %s due to errors", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "Summary: processed %d zip(s), touched %d student(s), copied %d file(s), skipped %d duplicate(s)",
        zips_processed,
        len(students_touched),
        files_copied,
        files_skipped,
    )

    finalize_result: Optional[FinalizeResult] = None
    if not encountered_errors:
        finalize_result = _finalize_success(base_dir, students_root)
        if finalize_result.zip_delete_failures or finalize_result.student_move_failures:
            encountered_errors = True

    if finalize_result:
        logger.info(
            "Post-processing: deleted %d zip(s), moved %d student folder(s) to %s",
            finalize_result.zips_deleted,
            finalize_result.students_moved,
            base_dir,
        )
        if finalize_result.zip_delete_failures:
            logger.error("Failed to delete %d zip(s); please clean up manually.", finalize_result.zip_delete_failures)
        if finalize_result.student_move_failures:
            logger.error(
                "Failed to move %d student folder(s); check permissions and resolve manually.",
                finalize_result.student_move_failures,
            )

    return 0 if not encountered_errors else 1


def _safe_extract(zip_file: ZipFile, destination: Path) -> None:
    """Extract zip contents into destination, guarding against path traversal."""
    dest_root = destination.resolve()
    for member in zip_file.infolist():
        member_path = dest_root.joinpath(member.filename).resolve()
        if not str(member_path).startswith(str(dest_root)):
            raise ValueError(f"Unsafe path detected in zip: {member.filename}")
        if member.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
        else:
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member) as src, member_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK_SIZE)


def _collapse_singleton_dir(path: Path) -> Path:
    """Collapse single directory layers to reach the content root."""
    current = path
    while True:
        entries = [
            entry
            for entry in current.iterdir()
            if entry.name not in NOISE_NAMES and not entry.name.startswith(".")
        ]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
            continue
        return current


def _split_filename(name: str) -> tuple[str, str]:
    """Split filename into base name and full suffix (handles multi-suffix names)."""
    path = Path(name)
    suffix = "".join(path.suffixes)
    if suffix:
        base = name[: -len(suffix)]
    else:
        base = name
    return base or name, suffix


def _files_identical(left: Path, right: Path) -> bool:
    """Check whether two files share the same size and SHA256 digest."""
    if left.stat().st_size != right.stat().st_size:
        return False
    return _file_digest(left) == _file_digest(right)


def _file_digest(path: Path) -> str:
    """Calculate SHA256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _gather_prefix_candidates(
    assignment_name: Optional[str], course_prefix: Optional[str], zip_stem: Optional[str]
) -> list[str]:
    """Build a list of possible prefixes to strip from student folder names."""
    candidates: list[str] = []
    for value, include_word_prefixes in (
        (assignment_name, True),
        (course_prefix, False),
        (zip_stem, False),
    ):
        if value:
            candidates.extend(_expand_prefix_variants(value, include_word_prefixes))
    # Preserve order while removing duplicates
    return [c for c in dict.fromkeys(candidates) if c]


def _expand_prefix_variants(value: str, include_word_prefixes: bool = False) -> list[str]:
    """Expand a value into reasonable textual variants for prefix stripping."""
    cleaned = value.strip()
    if not cleaned:
        return []

    normalized_space = re.sub(r"[\s_-]+", " ", cleaned).strip()
    variants = {cleaned, normalized_space}
    variants.add(normalized_space.replace(" ", "_"))
    variants.add(normalized_space.replace(" ", "-"))

    if include_word_prefixes:
        parts = cleaned.split()
        if parts:
            first_token = parts[0]
            variants.add(first_token)
            trimmed_first = first_token.rstrip("+-_.")
            if trimmed_first:
                variants.add(trimmed_first)
        words = normalized_space.split()
        for i in range(len(words), 0, -1):
            prefix = " ".join(words[:i])
            if len(prefix.replace(" ", "")) <= 1:
                continue
            variants.add(prefix)
            variants.add(prefix.replace(" ", "_"))
            variants.add(prefix.replace(" ", "-"))

    return [variant for variant in variants if variant]


def _strip_prefixes_from_left(name: str, prefixes: list[str]) -> str:
    """Strip known prefixes from the left side of a string."""
    unique_prefixes = sorted({prefix.strip() for prefix in prefixes if prefix}, key=len, reverse=True)
    while True:
        for prefix in unique_prefixes:
            match_end = _match_prefix(name, prefix)
            if match_end is None:
                continue
            if match_end < len(name):
                next_char = name[match_end]
                if next_char not in {" ", "_"}:
                    continue
            candidate = name[match_end:].lstrip(" _")
            if candidate and " - " in candidate:
                name = candidate
                break
        else:
            break
    return name.lstrip(" _-")


def _match_prefix(name: str, prefix: str) -> Optional[int]:
    """Return the end index of prefix match if present at the start of name."""
    pattern = re.compile(rf"^{re.escape(prefix)}", re.IGNORECASE)
    match = pattern.match(name)
    if match:
        return match.end()

    prefix_lower = prefix.lower()
    name_lower = name.lower()
    if name_lower.startswith(prefix_lower):
        return len(prefix)
    return None


def _finalize_success(base_dir: Path, students_root: Path) -> FinalizeResult:
    """Delete processed zip files and move student folders into the course root."""
    result = FinalizeResult()
    for zip_path in sorted(base_dir.glob("*.zip")):
        try:
            zip_path.unlink()
            result.zips_deleted += 1
        except OSError as exc:
            logger.error("Unable to delete zip file %s: %s", zip_path.name, exc)
            result.zip_delete_failures += 1

    if students_root.exists():
        student_dirs = [entry for entry in sorted(students_root.iterdir()) if entry.is_dir()]
        for student_dir in student_dirs:
            destination = _resolve_destination_dir(base_dir, student_dir.name)
            try:
                shutil.move(str(student_dir), str(destination))
                result.students_moved += 1
                if destination.name != student_dir.name:
                    logger.info("Moved %s to %s", student_dir.name, destination)
                else:
                    logger.info("Moved %s", destination.name)
            except OSError as exc:
                logger.error("Unable to move student folder %s: %s", student_dir.name, exc)
                result.student_move_failures += 1

        try:
            _remove_noise_entries(students_root)
            students_root.rmdir()
        except OSError:
            pass

    return result


def _resolve_destination_dir(root: Path, name: str) -> Path:
    """Resolve a destination directory path, avoiding name conflicts."""
    candidate = root / name
    if not candidate.exists():
        return candidate

    base, suffix = _split_filename(name)
    counter = 2
    while True:
        new_name = f"{base} ({counter}){suffix}"
        candidate = root / new_name
        if not candidate.exists():
            return candidate
        counter += 1


def _cleanup_empty_assignment_dir(assignment_dir: Path, students_root: Path) -> None:
    """Remove empty assignment and student directories created during processing."""
    if not assignment_dir.exists():
        return
    _remove_noise_entries(assignment_dir)
    try:
        next(assignment_dir.iterdir())
    except StopIteration:
        try:
            assignment_dir.rmdir()
        except OSError:
            return
        _cleanup_empty_parents(assignment_dir.parent, students_root)
    except FileNotFoundError:
        return


def _cleanup_empty_parents(path: Path, stop: Path) -> None:
    """Remove empty parent directories up to and including stop."""
    current = path
    while current is not None and current.exists():
        _remove_noise_entries(current)
        try:
            next(current.iterdir())
        except StopIteration:
            try:
                to_remove = current
                current = current.parent
                to_remove.rmdir()
            except OSError:
                return
            if to_remove == stop:
                return
            if current == stop.parent:
                return
            continue
        except FileNotFoundError:
            return
        break


def _remove_noise_entries(path: Path) -> None:
    """Remove known noise files and directories from the given path."""
    try:
        entries = list(path.iterdir())
    except FileNotFoundError:
        return
    for entry in entries:
        if entry.name not in NOISE_NAMES:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Organize Moodle submission zip files.")
    parser.add_argument(
        "base_directory",
        nargs="?",
        help="Path to the directory containing assignment zips (defaults to current working directory).",
    )
    return parser.parse_args(argv)


def run() -> int:
    """Entry point for CLI usage."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.base_directory:
        base_dir = Path(args.base_directory).expanduser()
    else:
        base_dir = Path.cwd()
    try:
        return main(base_dir)
    except Exception:
        logger.exception("Unhandled error during organization.")
        work_dir = base_dir / ".work_extract"
        logger.error("Temporary extraction directory left at %s", work_dir)
        return 1


if __name__ == "__main__":
    sys.exit(run())

# Example python3 organize_submissions.py ~/Downloads/EN-HM-2\ \(24-25\)