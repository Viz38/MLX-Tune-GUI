import { NextResponse } from 'next/server';
import { readdir, stat, readFile } from 'fs/promises';
import { join } from 'path';
import { homedir } from 'os';

export const dynamic = 'force-dynamic';

interface LocalModel {
  name: string;
  path: string;
  source: 'ollama' | 'lmstudio' | 'hf' | 'local';
}

async function scanOllama(): Promise<LocalModel[]> {
  const models: LocalModel[] = [];
  const base = join(homedir(), '.ollama', 'models', 'manifests', 'registry.ollama.ai', 'library');
  
  try {
    const registries = await readdir(base);
    for (const modelDir of registries) {
      const modelPath = join(base, modelDir);
      const s = await stat(modelPath);
      if (s.isDirectory()) {
        const tags = await readdir(modelPath);
        for (const tag of tags) {
          models.push({
            name: `${modelDir}:${tag}`,
            path: join(modelPath, tag), // Unique path for each model/tag
            source: 'ollama'
          });
        }
      }
    }
  } catch (e) {
    // Directory not found or inaccessible
  }
  return models;
}

async function scanHFHub(): Promise<LocalModel[]> {
  const models: LocalModel[] = [];
  const base = join(homedir(), '.cache', 'huggingface', 'hub');
  
  try {
    const dirs = await readdir(base);
    for (const dir of dirs) {
      if (dir.startsWith('models--')) {
        const parts = dir.split('--');
        const organization = parts[1];
        const name = parts.slice(2).join('--');
        const modelName = `${organization}/${name}`;
        
        const snapshotsDir = join(base, dir, 'snapshots');
        try {
          const snapshots = await readdir(snapshotsDir);
          if (snapshots.length > 0) {
            // Pick the first snapshot for now (usually only one)
            models.push({
              name: modelName,
              path: join(snapshotsDir, snapshots[0]),
              source: 'hf'
            });
          }
        } catch (e) {}
      }
    }
  } catch (e) {}
  return models;
}

async function scanLMStudio(): Promise<LocalModel[]> {
  const models: LocalModel[] = [];
  const paths = [
    join(homedir(), '.lmstudio', 'models'),
    join(homedir(), '.cache', 'lm-studio', 'models')
  ];
  
  for (const base of paths) {
    try {
      const providers = await readdir(base);
      for (const provider of providers) {
        const providerPath = join(base, provider);
        if ((await stat(providerPath)).isDirectory()) {
          const modelDirs = await readdir(providerPath);
          for (const mDir of modelDirs) {
            const mPath = join(providerPath, mDir);
            if ((await stat(mPath)).isDirectory()) {
              models.push({
                name: `${provider}/${mDir}`,
                path: mPath,
                source: 'lmstudio'
              });
            }
          }
        }
      }
    } catch (e) {}
  }
  return models;
}

export async function GET() {
  try {
    const [ollama, hf, lmstudio] = await Promise.all([
      scanOllama(),
      scanHFHub(),
      scanLMStudio()
    ]);

    return NextResponse.json({
      models: [...ollama, ...hf, ...lmstudio]
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
