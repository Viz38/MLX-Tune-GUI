import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // 1. Get Memory Stats (Unified Memory on Apple Silicon)
    const { stdout: vmStat } = await execAsync('vm_stat');
    
    // Parse vm_stat (page size is usually 4096)
    const pageSize = 4096;
    const lines = vmStat.split('\n');
    const getVal = (label: string) => {
      const line = lines.find(l => l.includes(label));
      return line ? parseInt(line.split(':')[1].trim()) : 0;
    };

    const free = getVal('Pages free');
    const inactive = getVal('Pages inactive');
    const speculative = getVal('Pages speculative');
    const wired = getVal('Pages wired down');
    const compressed = getVal('Pages occupied by compressor');

    // "Free" in macOS is more like free + inactive + speculative
    const memoryUsed = (wired + compressed) * pageSize;
    
    // 2. Get Total Memory
    const { stdout: memTotal } = await execAsync('sysctl -n hw.memsize');
    const totalMemBytes = parseInt(memTotal.trim());

    // 3. Get CPU Usage
    // We'll take a quick sample with top
    const { stdout: cpuInfo } = await execAsync('top -l 1 | grep "CPU usage"');
    // Format: "CPU usage: 10.00% user, 5.00% sys, 85.00% idle"
    const userMatch = cpuInfo.match(/(\d+\.\d+)% user/);
    const sysMatch = cpuInfo.match(/(\d+\.\d+)% sys/);
    const cpuUsage = (parseFloat(userMatch?.[1] || '0') + parseFloat(sysMatch?.[1] || '0')).toFixed(1);

    return NextResponse.json({
      memory: {
        total: totalMemBytes,
        used: memoryUsed,
        percent: ((memoryUsed / totalMemBytes) * 100).toFixed(1)
      },
      cpu: {
        percent: cpuUsage
      }
    });

  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
