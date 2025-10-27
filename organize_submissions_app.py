import logging
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, List, Optional

import organize_submissions


class TextQueueHandler(logging.Handler):
    """Logging handler that pushes formatted records into a queue."""

    def __init__(self, log_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        self.log_queue.put(message)


class SubmissionOrganizerApp:
    """Tkinter GUI to orchestrate Moodle submission organization."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("Moodle Submission Organizer")
        self.master.geometry("800x500")

        self.selected_zip_paths: List[Path] = []
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.log_handler = TextQueueHandler(self.log_queue)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))

        self._build_widgets()
        self._poll_log_queue()

        self.processing_thread: Optional[threading.Thread] = None

    def _build_widgets(self) -> None:
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.master, padding=12)
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        course_frame = ttk.Frame(main_frame)
        course_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        course_frame.columnconfigure(1, weight=1)

        ttk.Label(course_frame, text="Course folder:").grid(row=0, column=0, sticky="w")

        self.course_var = tk.StringVar()
        course_entry = ttk.Entry(course_frame, textvariable=self.course_var, width=60)
        course_entry.grid(row=0, column=1, padx=8, sticky="ew")

        self.browse_button = ttk.Button(course_frame, text="Browse...", command=self._choose_course_folder)
        self.browse_button.grid(row=0, column=2, sticky="e")

        zips_frame = ttk.LabelFrame(main_frame, text="Selected ZIP files", padding=(8, 4))
        zips_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")
        zips_frame.columnconfigure(0, weight=1)

        self.zip_list_var = tk.StringVar(value=[])
        self.zip_listbox = tk.Listbox(
            zips_frame,
            listvariable=self.zip_list_var,
            height=6,
            selectmode=tk.SINGLE,
        )
        self.zip_listbox.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)

        self.select_button = ttk.Button(buttons_frame, text="Select ZIP files...", command=self._select_zip_files)
        self.select_button.grid(row=0, column=0, sticky="w")

        self.clear_button = ttk.Button(buttons_frame, text="Clear Selection", command=self._clear_zip_selection)
        self.clear_button.grid(row=0, column=1, sticky="w", padx=8)

        run_button = ttk.Button(buttons_frame, text="Organize Submissions", command=self._start_processing)
        run_button.grid(row=0, column=2, sticky="e")
        self.run_button = run_button

        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=6)

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=(8, 4))
        log_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _choose_course_folder(self) -> None:
        initial = Path(self.course_var.get()).expanduser() if self.course_var.get() else Path.cwd()
        selected = filedialog.askdirectory(parent=self.master, initialdir=initial)
        if selected:
            self.course_var.set(selected)

    def _select_zip_files(self) -> None:
        initial = Path(self.course_var.get()).expanduser() if self.course_var.get() else Path.cwd()
        selected = filedialog.askopenfilenames(
            parent=self.master,
            title="Select assignment ZIP files",
            initialdir=initial,
            filetypes=[("ZIP archives", "*.zip"), ("All files", "*.*")],
        )
        if not selected:
            return

        self.selected_zip_paths = [Path(path) for path in selected]
        self._refresh_zip_listbox()

    def _clear_zip_selection(self) -> None:
        self.selected_zip_paths = []
        self._refresh_zip_listbox()

    def _refresh_zip_listbox(self) -> None:
        display = [str(path) for path in self.selected_zip_paths]
        self.zip_list_var.set(display)

    def _start_processing(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Organizer", "Processing is already running.", parent=self.master)
            return

        course_folder = self.course_var.get().strip()
        if not course_folder:
            messagebox.showwarning("Organizer", "Please enter or select a course folder.", parent=self.master)
            return

        if not self.selected_zip_paths:
            messagebox.showwarning("Organizer", "Please select at least one ZIP file.", parent=self.master)
            return

        self._set_status("Preparing...")
        self._set_running_state(True)
        self._clear_log()
        self._log_message("Starting organisation...")

        self.processing_thread = threading.Thread(target=self._process_submission, daemon=True)
        self.processing_thread.start()

    def _process_submission(self) -> None:
        course_path = Path(self.course_var.get()).expanduser()
        zip_paths = list(self.selected_zip_paths)

        try:
            self._log_message(f"Course folder: {course_path}")
            course_path.mkdir(parents=True, exist_ok=True)

            prepared_zips = self._prepare_zip_files(zip_paths, course_path)
            if not prepared_zips:
                raise ValueError("No valid ZIP files were provided.")

            root_logger = logging.getLogger()
            module_logger = organize_submissions.logger
            original_root_level = root_logger.level
            original_module_level = module_logger.level

            capture_handlers = self._attach_logging_handlers()

            try:
                root_logger.setLevel(logging.INFO)
                module_logger.setLevel(logging.INFO)

                exit_code = organize_submissions.main(course_path)
                if exit_code == 0:
                    self._log_message("Organisation completed successfully.")
                    message = "Submissions organised successfully."
                    self._notify_user(message, success=True)
                else:
                    self._log_message("Organisation completed with issues. Check the log for details.")
                    message = "Submissions organised with some issues. See log for details."
                    self._notify_user(message, success=False)
            finally:
                root_logger.setLevel(original_root_level)
                module_logger.setLevel(original_module_level)
                self._detach_logging_handlers(capture_handlers)

        except Exception as exc:
            self._log_message(f"Error: {exc}")
            self._notify_user(f"An error occurred:\n{exc}", success=False)
        finally:
            self.master.after(0, lambda: self._set_running_state(False))
            self.master.after(0, lambda: self._set_status("Idle"))

    def _prepare_zip_files(self, zip_paths: Iterable[Path], course_path: Path) -> List[Path]:
        """Ensure selected ZIP files exist inside the course folder."""
        prepared: List[Path] = []
        for src in zip_paths:
            if not src.exists():
                self._log_message(f"Skipping missing ZIP file: {src}")
                continue

            destination = course_path / src.name

            try:
                if destination.exists():
                    if self._files_are_identical(src, destination):
                        self._log_message(f"Using existing ZIP file: {destination}")
                        prepared.append(destination)
                        continue
                    else:
                        destination = self._resolve_conflict_path(destination)
                        self._log_message(f"Copying to avoid name conflict: {destination}")

                if src.resolve() == destination.resolve():
                    self._log_message(f"ZIP already in course folder: {destination}")
                    prepared.append(destination)
                    continue

                shutil.copy2(src, destination)
                self._log_message(f"Copied {src.name} to course folder.")
                prepared.append(destination)
            except OSError as exc:
                self._log_message(f"Failed to prepare {src}: {exc}")

        return prepared

    def _files_are_identical(self, first: Path, second: Path) -> bool:
        try:
            first_stat = first.stat()
            second_stat = second.stat()
            return first_stat.st_size == second_stat.st_size and first_stat.st_mtime_ns == second_stat.st_mtime_ns
        except OSError:
            return False

    def _resolve_conflict_path(self, path: Path) -> Path:
        base = path.stem
        suffix = path.suffix
        counter = 2
        candidate = path
        while candidate.exists():
            candidate = path.with_name(f"{base} ({counter}){suffix}")
            counter += 1
        return candidate

    def _attach_logging_handlers(self) -> List[logging.Handler]:
        root_logger = logging.getLogger()
        capture_handlers: List[logging.Handler] = [self.log_handler]
        root_logger.addHandler(self.log_handler)
        organize_submissions.logger.addHandler(self.log_handler)
        return capture_handlers

    def _detach_logging_handlers(self, handlers: Iterable[logging.Handler]) -> None:
        root_logger = logging.getLogger()
        for handler in handlers:
            try:
                root_logger.removeHandler(handler)
            except ValueError:
                pass
            try:
                organize_submissions.logger.removeHandler(handler)
            except ValueError:
                pass

    def _notify_user(self, message: str, success: bool) -> None:
        self.master.after(
            0,
            lambda: (
                messagebox.showinfo("Organizer", message, parent=self.master)
                if success
                else messagebox.showwarning("Organizer", message, parent=self.master)
            ),
        )

    def _poll_log_queue(self) -> None:
        try:
            while True:
                message = self.log_queue.get_nowait()
                self._append_log_text(message)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self._poll_log_queue)

    def _append_log_text(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_status(self, status: str) -> None:
        self.master.after(0, lambda: self.status_var.set(status))

    def _set_running_state(self, running: bool) -> None:
        def _apply_state() -> None:
            state = tk.DISABLED if running else tk.NORMAL
            for widget in (self.run_button, self.select_button, self.clear_button, self.browse_button):
                widget.configure(state=state)

        self.master.after(0, _apply_state)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log_message(self, message: str) -> None:
        self.log_queue.put(message)


def main() -> None:
    root = tk.Tk()
    SubmissionOrganizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
