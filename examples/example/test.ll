; ModuleID = 'test'

declare i32 @start(i8*, i8*)
declare i32 @lumen_print(i8*, i8*)

@.app_name = private global [5 x i8] c"test\00"
@.app_version = private global [6 x i8] c"0.1.0\00"

define i32 @main(i32 %argc, i8* %argv) {
    ; Initialize globals
    ;store i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i32 0, i32 0), i8** @APP_NAME
    ;store i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i32 0, i32 0), i8** @APP_VERSION
    ; Start runtime
    ;%1 = getelementptr inbounds ([5 x i8], [5 x i8]* @APP_NAME, i32 0, i32 0)
    %1 = bitcast [5 x i8]* @.app_name to i8*
    %2 = bitcast [6 x i8]* @.app_version to i8*
    %res = call i32 @start(i8* %1, i8* %2)
    ret i32 %res
}
