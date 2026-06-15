@echo off

echo Uninstalling cosensim...
pip uninstall cosensim -y

echo Installing from parent directory...
pip install ..

echo Removing local build directory...
if exist .\build (
    rmdir /S /Q .\build
)

echo Removing gh-pages branch build directory...
if exist ..\..\cosensim.gh-pages\*.html (
    rmdir /S /Q ..\..\cosensim.gh-pages\_sources
    rmdir /S /Q ..\..\cosensim.gh-pages\_static
    del /S /Q /F ..\..\cosensim.gh-pages\*.html
    del /S /Q /F ..\..\cosensim.gh-pages\*.buildinfo
    del /S /Q /F ..\..\cosensim.gh-pages\*.inv
    del /S /Q /F ..\..\cosensim.gh-pages\*.js
)


echo Building HTML...
call make html
if %ERRORLEVEL% neq 0 (
    echo Error building HTML. Exiting...
    exit /b %ERRORLEVEL%
)

echo Moving build directory...
if exist .\build (
    xcopy .\build\html\_sources ..\..\cosensim.gh-pages\_sources /E /Y /I
    xcopy .\build\html\_static ..\..\cosensim.gh-pages\_static /E /Y /I
    copy .\build\html\* ..\..\cosensim.gh-pages\
)




echo Done!
pause
