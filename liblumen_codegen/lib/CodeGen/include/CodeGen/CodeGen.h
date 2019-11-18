#ifndef LUMEN_CODEGEN_H
#define LUMEN_CODEGEN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OpaqueCodeGenContext *CodeGenContext;

CodeGenContext CodeGenContextCreate(int argc, char **argv);
void CodeGenContextDispose(CodeGenContext ctx);

#ifdef __cplusplus
} // end extern
#endif

#endif // end LUMEN_CODEGEN_H