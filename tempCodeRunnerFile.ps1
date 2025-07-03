# auto_sync.ps1
cd "C:\Users\Asus Vivobook 15\Desktop\CSI"

# Add all changed files
git add .

# Commit with current date and time
$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Auto-sync on $date"

# Push to GitHub
git push origin main
