#ifndef LUMEN_SUPPORT_ERRORS_H
#define LUMEN_SUPPORT_ERRORS_H

extern "C" {
char *LLVMLumenGetLastError(void);
void LLVMLumenSetLastError(const char *);
void LLVMLumenInstallFatalErrorHandler();
}

#endif
