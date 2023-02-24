#pragma once

#if defined(_WIN32)

#include "windows.h"
#include "llvm/Support/raw_ostream.h"

class raw_win32_handle_ostream : public raw_pwrite_stream {
  HANDLE Handle;
  bool ShouldClose;
  bool SupportsSeeking = false;
  bool ColorEnabled = true;
  bool IsConsole = false;

  std::error_code EC;

  uint64_t pos = 0;
  /// See raw_ostream::write_impl.

  void write_impl(const char *Ptr, size_t Size) override;
  void pwrite_impl(const char *Ptr, size_t Size, uint64_t Offset) override;

  /// Return the current position within the stream, not counting the bytes
  /// currently in the buffer.
  uint64_t current_pos() const override { return pos; }

  /// Determine an efficient buffer size.
  size_t preferred_buffer_size() const override;

  /// Set the flag indicating that an output error has been encountered.
  void error_detected(std::error_code EC) { this->EC = EC; }

  void anchor() override;

public:
  raw_win32_handle_ostream(HANDLE h, bool shouldClose, bool unbuffered = false);

  ~raw_win32_handle_ostream() override;

  void close();

  bool supportsSeeking() { return SupportsSeeking; }

  uint64_t seek(uint64_t off);

  raw_ostream &changeColor(enum Colors colors, bool bold = false,
                           bool bg = false) override;
  raw_ostream &resetColor() override;
  raw_ostream &reverseColor() override;

  bool is_displayed() const override;
  bool has_colors() const override;
  void enable_colors(bool enable) override { ColorEnabled = enable; }

  std::error_code error() const { return EC; }

  bool has_error() const { return bool(EC); }

  void clear_error() { EC = std::error_code(); }
};

#endif // defined(_WIN32)
