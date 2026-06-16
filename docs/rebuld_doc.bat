@echo off

echo Uninstalling coinfosim...
pip uninstall coinfosim -y

echo Installing from parent directory...
pip install ..

echo Removing local build directory...
if exist .\build (
    rmdir /S /Q .\build
)

echo Removing gh-pages branch build directory...
if exist ..\..\coinfosim.gh-pages\*.html (
    rmdir /S /Q ..\..\coinfosim.gh-pages\_sources
    rmdir /S /Q ..\..\coinfosim.gh-pages\_static
    del /S /Q /F ..\..\coinfosim.gh-pages\*.html
    del /S /Q /F ..\..\coinfosim.gh-pages\*.buildinfo
    del /S /Q /F ..\..\coinfosim.gh-pages\*.inv
    del /S /Q /F ..\..\coinfosim.gh-pages\*.js
)


echo Building HTML...
call make html
if %ERRORLEVEL% neq 0 (
    echo Error building HTML. Exiting...
    exit /b %ERRORLEVEL%
)

echo Moving build directory...
if exist .\build (
    xcopy .\build\html\_sources ..\..\coinfosim.gh-pages\_sources /E /Y /I
    xcopy .\build\html\_static ..\..\coinfosim.gh-pages\_static /E /Y /I
    copy .\build\html\* ..\..\coinfosim.gh-pages\
)




echo Done!
pause
