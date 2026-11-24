# Prints one JSON line per sample with the memory of a Chrome instance's process tree (Windows only).
# The instance is the browser process whose command line contains $Profile; its descendants are summed.
# peakRenderer and peakGpuProcess are the largest PeakPageFileUsage of each kind: Windows tracks it for each
# process since it started, so it does not depend on when the samples fall.
# GPU memory comes from the "GPU Process Memory" counters, which cover the NVIDIA and the Intel GPU.
param([Parameter(Mandatory)][string]$Profile, [int]$IntervalMs = 250)

$ErrorActionPreference = "SilentlyContinue"
while ($true) {
  $all = Get-CimInstance Win32_Process -Filter "Name='chrome.exe' OR Name='msedge.exe'"
  $root = $all | Where-Object { $_.CommandLine -like "*$Profile*" -and $_.CommandLine -notlike "*--type=*" } | Select-Object -First 1
  $pids = @{}
  if ($root) {
    $pids[[int]$root.ProcessId] = $true
    do {
      $added = $false
      foreach ($p in $all) {
        if ($pids.ContainsKey([int]$p.ParentProcessId) -and -not $pids.ContainsKey([int]$p.ProcessId)) {
          $pids[[int]$p.ProcessId] = $true
          $added = $true
        }
      }
    } while ($added)
  }
  $private = 0; $maxPrivate = 0; $renderer = 0; $gpuProcess = 0; $peakRenderer = 0; $peakGpuProcess = 0
  foreach ($p in $all) {
    if ($pids.ContainsKey([int]$p.ProcessId)) {
      $bytes = [int64]$p.PrivatePageCount
      $private += $bytes
      $peak = [int64]$p.PeakPageFileUsage * 1024  # WMI gives KB
      if ($bytes -gt $maxPrivate) { $maxPrivate = $bytes }
      if ($p.CommandLine -like "*--type=renderer*") {
        $renderer += $bytes
        if ($peak -gt $peakRenderer) { $peakRenderer = $peak }
      }
      if ($p.CommandLine -like "*--type=gpu-process*") {
        $gpuProcess += $bytes
        if ($peak -gt $peakGpuProcess) { $peakGpuProcess = $peak }
      }
    }
  }
  $dedicated = 0; $shared = 0
  $counters = (Get-Counter "\GPU Process Memory(*)\Dedicated Usage", "\GPU Process Memory(*)\Shared Usage").CounterSamples
  foreach ($c in $counters) {
    if ($c.InstanceName -match '^pid_(\d+)_' -and $pids.ContainsKey([int]$Matches[1])) {
      if ($c.Path -like "*dedicated usage") { $dedicated += $c.CookedValue } else { $shared += $c.CookedValue }
    }
  }
  $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  Write-Output ("{{""t"":{0},""procs"":{1},""private"":{2},""maxPrivate"":{3},""gpuDedicated"":{4},""gpuShared"":{5},""renderer"":{6},""gpuProcess"":{7},""peakRenderer"":{8},""peakGpuProcess"":{9}}}" -f $t, $pids.Count, $private, $maxPrivate, [int64]$dedicated, [int64]$shared, $renderer, $gpuProcess, $peakRenderer, $peakGpuProcess)
  Start-Sleep -Milliseconds $IntervalMs
}
