# Lumen.OTP.Log.Parser

Parses the log from `cargo test`

1. `cd ../../../..`
2. Run tests and put ANSI-stripped output to `test.log`: `cargo make test -- --package liblumen_otp lumen::otp:: 2&>1 | sed 's/\x1b\[[0-9;]*m//g' | tee test.log`  
1. `cd native_implemented/otp/tests/log_parser`
3. Parse those log into CSV: `mix lumen.otp.log.parse ../../../../test.log > test.log.csv`
