@echo off

echo Uninstalling slacgs...
pip uninstall slacgs -y

echo Installing from parent directory...
pip install ..

echo Removing local build directory...
if exist .\build (
    rmdir /S /Q .\build
)

echo Removing gh-pages branch build directory...
if exist ..\..\slacgs.gh-pages\*.html (
    rmdir /S /Q ..\..\slacgs.gh-pages\_sources
    rmdir /S /Q ..\..\slacgs.gh-pages\_static
    del /S /Q /F ..\..\slacgs.gh-pages\*.html
    del /S /Q /F ..\..\slacgs.gh-pages\*.buildinfo
    del /S /Q /F ..\..\slacgs.gh-pages\*.inv
    del /S /Q /F ..\..\slacgs.gh-pages\*.js
)


echo Building HTML...
call make html
if %ERRORLEVEL% neq 0 (
    echo Error building HTML. Exiting...
    exit /b %ERRORLEVEL%
)

echo Moving build directory...
if exist .\build (
    xcopy .\build\html\_sources ..\..\slacgs.gh-pages\_sources /E /Y /I
    xcopy .\build\html\_static ..\..\slacgs.gh-pages\_static /E /Y /I
    copy .\build\html\* ..\..\slacgs.gh-pages\
)




echo Done!
pause
