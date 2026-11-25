[CmdletBinding()]
param([string] $FromCheckout)

$ErrorActionPreference = 'Stop'
$relativePath = 'crates/g6-formats/src/bcpack-tables.bin'
$destination = Join-Path $PSScriptRoot $relativePath
$expectedHash = '1F1D4760419202BB5672F0C428CE51E456B5D08FBAEC8F371D043FD94E01C36B'
# The source address is kept in CREDITS.md; the pinned SHA-256 above guards its content.
$credits = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'CREDITS.md'))
$address = [regex]::Match($credits, '(?m)^Format tables address: (https://raw\.githubusercontent\.com/\S+/bcpack-tables\.bin)\s*$')
if (!$address.Success) { throw 'CREDITS.md does not give the format tables address.' }
$sourceUrl = $address.Groups[1].Value
# The path inside the source repository, which keeps its own folder names.
$upstreamPath = ($sourceUrl -split '/', 7)[6]

function Assert-FormatTableHash([string] $Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha.ComputeHash([IO.File]::ReadAllBytes($Path))
        $actual = [BitConverter]::ToString($digest).Replace('-', '')
    } finally { $sha.Dispose() }
    if ($actual -ne $expectedHash) {
        throw "Unexpected BCPack table SHA-256 for '$Path'. Expected $expectedHash, received $actual."
    }
}

if (Test-Path -LiteralPath $destination) {
    Assert-FormatTableHash $destination
    Write-Output 'Verified existing BCPack format tables.'
    return
}

$tempFile = Join-Path (Split-Path -Parent $destination) ([IO.Path]::GetRandomFileName())
try {
    if ($PSBoundParameters.ContainsKey('FromCheckout')) {
        $source = Join-Path $FromCheckout $upstreamPath
        Copy-Item -LiteralPath $source -Destination $tempFile
    } else {
        Invoke-WebRequest -Uri $sourceUrl -OutFile $tempFile -UseBasicParsing
    }
    Assert-FormatTableHash $tempFile
    [IO.File]::Move($tempFile, $destination)
    Write-Output 'Verified and installed the BCPack format tables.'
} finally {
    if (Test-Path -LiteralPath $tempFile) {
        Remove-Item -LiteralPath $tempFile
    }
}
