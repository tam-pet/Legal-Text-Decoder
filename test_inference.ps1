# Test Inference Script
# Gyorsan tesztelheted a tanított modelleket

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  LEGAL TEXT DECODER - Inference Demo" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Set Python path
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$ProjectDir\src;$env:PYTHONPATH"

# Test texts
$testTexts = @(
    "A szolgáltatás használatához internetkapcsolat szükséges.",
    "A Szolgáltató fenntartja a jogot, hogy a szolgáltatást bármikor módosítsa vagy megszüntesse, előzetes értesítés nélkül.",
    "A jelen ÁSZF 12.3. pontjában meghatározott, a Ptk. 6:78. § (2) bekezdése szerinti elállási jog gyakorlásának határideje a termék átvételétől számított 14 nap.",
    "Az adatkezelési szabályzat megtalálható a www.example.com weboldalon.",
    "Amennyiben a Felhasználó nem ért egyet a módosításokkal, úgy jogosult a szerződést 30 napon belül felmondani."
)

Write-Host "[INFO] Tesztelendő szövegek száma: $($testTexts.Count)" -ForegroundColor Yellow
Write-Host ""

foreach ($i in 0..($testTexts.Count - 1)) {
    $text = $testTexts[$i]
    $num = $i + 1
    
    Write-Host "[$num/$($testTexts.Count)] " -ForegroundColor Cyan -NoNewline
    Write-Host "Szöveg: " -NoNewline
    if ($text.Length -gt 70) {
        Write-Host "$($text.Substring(0, 70))..." -ForegroundColor White
    } else {
        Write-Host "$text" -ForegroundColor White
    }
    
    # Run inference
    $output = & python src\a04_inference.py --text $text 2>&1 | Out-String
    
    # Parse result (extract rating from JSON output)
    if ($output -match '"prediction":\s*(\d+)') {
        $rating = $Matches[1]
        $description = switch ($rating) {
            "1" { "Nagyon nehezen vagy nem értelmezhető" }
            "2" { "Nehezen értelmezhető" }
            "3" { "Valamennyire érthető" }
            "4" { "Végigolvasva megértem" }
            "5" { "Könnyen, egyből érthető" }
        }
        
        Write-Host "         Értékelés: " -NoNewline -ForegroundColor Yellow
        
        # Color code the rating
        $color = switch ($rating) {
            "1" { "Red" }
            "2" { "DarkYellow" }
            "3" { "Yellow" }
            "4" { "Green" }
            "5" { "Cyan" }
        }
        
        Write-Host "$rating/5" -ForegroundColor $color -NoNewline
        Write-Host " - $description" -ForegroundColor Gray
    } else {
        Write-Host "         [ERROR] Nem sikerült értékelni" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  Demo befejezve!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline; Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "Kipróbálhatod saját szöveggel is:" -ForegroundColor Yellow
Write-Host "  python src\a04_inference.py --interactive" -ForegroundColor Cyan
Write-Host ""
