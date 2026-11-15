@echo off
:: Run the internal launcher.
:: You can also specify --host and --port, for example:  ... gradio-internal.bat" --host 127.0.0.1 --port 5555
call "%~dp0..\tools\gradio-internal.bat" %*
