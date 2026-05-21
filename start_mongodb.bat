@echo off
echo ========================================
echo   Demarrage MongoDB pour NovaClinic
echo ========================================
echo.

REM Demarrer MongoDB en tant qu'administrateur
echo [1/2] Demarrage du service MongoDB...
net start MongoDB

if %errorlevel% neq 0 (
    echo.
    echo ERREUR: Impossible de demarrer MongoDB
    echo Veuillez executer ce script en tant qu'administrateur
    echo.
    echo Clic droit sur le fichier ^> Executer en tant qu'administrateur
    echo.
    pause
    exit /b 1
)

echo.
echo [2/2] Verification de la connexion...
timeout /t 3 /nobreak >nul

mongo --eval "db.version()" >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   MongoDB demarre avec succes!
    echo   URL: mongodb://localhost:27017
    echo ========================================
    echo.
) else (
    echo.
    echo ATTENTION: MongoDB demarre mais la connexion n'est pas encore prete
    echo Attendez quelques secondes et verifiez avec MongoDB Compass
    echo.
)

pause
