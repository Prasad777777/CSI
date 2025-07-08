# auto_sync.ps1
cd "C:\Users\Asus Vivobook 15\Desktop\CSI"

# Add changes
git add .

# Commit only if there are changes
if (-not (git diff --cached --quiet)) {
    $date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto-sync on $date"
    git push origin main
} else {
    Write-Output "✅ No changes to commit."
}




                                                                                                                                                                                                  