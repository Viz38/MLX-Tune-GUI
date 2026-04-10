import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

export async function POST(request: Request) {
  try {
    const { action, modelName, hubUsername } = await request.json();
    
    let pythonCode = '';
    const outputDir = 'outputs'; // Default
    
    if (action === 'hub') {
      const repoId = `${hubUsername}/${modelName.split('/').pop()}-mlx`;
      pythonCode = `
from mlx_tune import FastLanguageModel
import os

print(f"Loading model for Hub Push: {modelName}", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained("${modelName}")

print(f"Pushing to Hugging Face Hub: {repoId}", flush=True)
model.push_to_hub("${repoId}", tokenizer=tokenizer)
print("Push completed successfully!", flush=True)
`;
    } else if (action === 'gguf') {
      const ggufPath = join(outputDir, 'gguf_model');
      pythonCode = `
from mlx_tune import FastLanguageModel
import os

print(f"Loading model for GGUF Export: {modelName}", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained("${modelName}")

print(f"Exporting to GGUF: {ggufPath}", flush=True)
if not os.path.exists("${outputDir}"):
    os.makedirs("${outputDir}")
    
model.save_pretrained_gguf("${ggufPath}", tokenizer=tokenizer)
print("GGUF Export completed successfully!", flush=True)
`;
    }

    const scriptPath = join(tmpdir(), `mlx_tune_export_${Date.now()}.py`);
    await writeFile(scriptPath, pythonCode);
    
    const child = spawn('python3', [scriptPath], {
      cwd: join(process.cwd(), '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    const stream = new ReadableStream({
      start(controller) {
        child.stdout.on('data', (data) => controller.enqueue(data));
        child.stderr.on('data', (data) => controller.enqueue(data));
        child.on('close', (code) => {
          controller.enqueue(new TextEncoder().encode(`\nProcess finished with code ${code}\n`));
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
      },
    });

  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
