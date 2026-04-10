import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { generatePythonScript } from '@/lib/scriptGenerator';
import { MLXTuneConfig } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const config: MLXTuneConfig = await request.json();
    
    // Generate the python code based on config
    const pythonCode = generatePythonScript(config);
    
    const scriptPath = join(tmpdir(), `mlx_tune_run_${Date.now()}.py`);
    await writeFile(scriptPath, pythonCode);
    
    // Spawn the python process
    // Run from workspace root to use the mlx-tune clone correctly, or rely on system python
    // Assumes mlx-tune is installed or available in the path. If running local clone, cwd matters.
    const child = spawn('python3', [scriptPath], {
      // By default Next.js server starts where we ran npm run dev, likely web folder.
      // We'll set the cwd explicitly to parent folder if it's the repo root.
      cwd: join(process.cwd(), '..'), 
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(`Starting ${config.type} workflow...\n`));
        controller.enqueue(new TextEncoder().encode(`Executing script: ${scriptPath}\n\n`));
        
        child.stdout.on('data', (data) => {
          controller.enqueue(data);
        });
        
        child.stderr.on('data', (data) => {
          controller.enqueue(data);
        });
        
        child.on('close', (code) => {
          controller.enqueue(new TextEncoder().encode(`\nProcess exited with code ${code}\n`));
          controller.close();
        });
        
        child.on('error', (err) => {
          controller.enqueue(new TextEncoder().encode(`\nError: ${err.message}\n`));
          controller.close();
        });
      },
      cancel() {
        child.kill();
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
        'Cache-Control': 'no-cache, no-transform',
      },
    });

  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
