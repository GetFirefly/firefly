#if defined(_WIN32)

#include "firefly/llvm/raw_win32_handle_ostream.h"

#include "Windows/WindowsSupport.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"

raw_win32_handle_ostream::raw_win32_handle_ostream(HANDLE h, bool isStdIo,
                                                   bool shouldClose,
                                                   bool unbuffered)
    : raw_pwrite_stream(unbuffered), Handle(h), ShouldClose(shouldClose) {
  IsConsole = ::GetFileType(Handle) == FILE_TYPE_CHAR;
  if (IsConsole) {
    ShouldClose = false;
    return;
  }

  // Get the current position
  DWORD loc =
      ::SetFilePointer(Handle, (LONG)0, nullptr,
                       FILE_CURRENT) if (loc == INVALID_SET_FILE_POINTER) {
    SupportsSeeking = false;
    pos = 0;
  }
  else {
    pos = static_cast<uint64_t>(loc);
  }
}

raw_win32_handle_ostream::~raw_win32_handle_ostream() {
  if (Handle != (HANDLE) nullptr) {
    flush();
    if (ShouldClose) {
      if (::CloseHandle(Handle)) {
        EC = std::error_code(::GetLastError());
        error_detected(EC);
      }
    }
  }

  if (has_error()) {
    report_fatal_error("IO failure on output stream: " + error().message(),
                       /*gen_crash_diag=*/false);
  }
}

static bool write_console_impl(HANDLE handle, StringRef data) {
  SmallVector<wchar_t, 256> wideText;

  if (auto ec = sys::windows::UTF8ToUTF16(data, wideText)) return false;

  // On Windows 7 and earlier, WriteConsoleW has a low maximum amount of data
  // that can be written to the console at a time.
  size_t maxWriteSize = wideText.size();
  if (!RunningWindows8OrGreater()) maxWriteSize = 32767;

  size_t wCharsWritten = 0;
  do {
    size_t wCharsToWrite =
        std::min(maxWriteSize, wideText.size() - wCharsWritten);
    DWORD actuallyWritten;
    bool success =
        ::WriteConsoleW(handle, &wideText[wCharsWritten], wCharsToWrite,
                        &actuallyWritten, /*reserved=*/nullptr);

    // The most likely reason to fail is that the handle does not point to a
    // console, fall back to write
    if (!success) return false;

    wCharsWritten += actuallyWritten;
  } while (wCharsWritten != wideText.size());
  return true;
}

void raw_win32_handle_ostream::write_impl(const char *ptr, size_t size) {
  assert(Handle != (HANDLE) nullptr && "File already closed.");
  pos += size;

  if (IsConsole)
    if (write_console_impl(Handle, StringRef(ptr, size))) return;

  DWORD bytesWritten = 0;
  bool pending = true;
  do {
    if (!::WriteFile(Handle, (LPCVOID)ptr, (DWORD)size, &bytesWritten,
                     nullptr)) {
      auto err = ::GetLastError();
      // Async write
      if (err == ERROR_IO_PENDING) {
        continue;
      } else {
        EC = std::error_code(err);
        error_detected(EC);
        break;
      }
    } else {
      pending = false;
    }
  } while (pending);
}

void raw_win32_handle_ostream::close() {
  assert(ShouldClose);
  ShouldClose = false;
  flush();
  if (::CloseHandle(Handle)) {
    EC = std::error_code(::GetLastError());
    error_detected(EC);
  }
  Handle = (HANDLE) nullptr;
}

uint64_t raw_win32_handle_ostream::seek(uint64_t off) {
  assert(SupportsSeeking && "Stream does not support seeking!");
  flush();
  LARGE_INTEGER li;
  li.QuadPart = off;
  li.LowPart = ::SetFilePointer(Handle, li.LowPart, &li.HighPart, FILE_BEGIN);
  if (li.LowPart == INVALID_SET_FILE_POINTER) {
    auto err = ::GetLastError();
    if (err != NO_ERROR) {
      pos = (uint64_t)-1;
      li.QuadPart = -1;
      error_detected(err);
      return pos;
    }
  }
  pos = li.QuadPart;
  return pos;
}

void raw_win32_handle_ostream::pwrite_impl(const char *ptr, size_t size,
                                           uint64_t offset) {
  uint64_t position = tell();
  seek(offset);
  write(ptr, size);
  seek(position);
}

size_t raw_win32_handle_ostream::preferred_buffer_size() const {
  if (IsConsole) return 0;
  return raw_ostream::preferred_buffer_size();
}

raw_ostream &raw_win32_handle_ostream::changeColor(enum Colors colors,
                                                   bool bold, bool bg) {
  if (!ColorEnabled) return *this;

  if (sys::Process::ColorNeedsFlush()) flush();
  const char *colorcode =
      (colors == SAVEDCOLOR)
          ? sys::Process::OutputBold(bg)
          : sys::Process::OutputColor(static_cast<char>(colors), bold, bg);
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

raw_ostream &raw_win32_handle_ostream::resetColor() {
  if (!ColorEnabled) return *this;

  if (sys::Process::ColorNeedsFlush()) flush();
  const char *colorcode = sys::Process::ResetColor();
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

raw_ostream &raw_win32_handle_ostream::reverseColor() {
  if (!ColorEnabled) return *this;

  if (sys::Process::ColorNeedsFlush()) flush();
  const char *colorcode = sys::Process::OutputReverse();
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

bool raw_win32_handle_ostream::is_displayed() const { return IsConsole; }

bool raw_win32_handle_ostream::has_colors() const { return ColorEnabled; }

void raw_win32_handle_ostream::anchor() {}

#endif
