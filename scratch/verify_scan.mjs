import { readdir, stat } from 'fs/promises';
import { join } from 'path';
import { homedir } from 'os';

async function scanOllama() {
  const modelPath = join(homedir(), '.ollama', 'models', 'manifests', 'registry.ollama.ai', 'library');
  try {
    const entries = await readdir(modelPath);
    console.log('Ollama folders found:', entries);
    for (const modelDir of entries) {
      const fullPath = join(modelPath, modelDir);
      if ((await stat(fullPath)).isDirectory()) {
         const tags = await readdir(fullPath);
         console.log(`Tags for ${modelDir}:`, tags);
         for (const tag of tags) {
           console.log(` - Entry: ${modelDir}:${tag} -> ${join(fullPath, tag)}`);
         }
      }
    }
  } catch (e) {
    console.log('Ollama path not found or error:', e.message);
  }
}

async function scanLMStudio() {
  const paths = [
    join(homedir(), '.lmstudio', 'models'),
    join(homedir(), '.cache', 'lm-studio', 'models')
  ];
  for (const base of paths) {
    console.log(`Scanning LM Studio Path: ${base}`);
    try {
      const providers = await readdir(base);
      for (const provider of providers) {
        const providerPath = join(base, provider);
        if ((await stat(providerPath)).isDirectory()) {
          const modelDirs = await readdir(providerPath);
          for (const mDir of modelDirs) {
            console.log(` - LMStudio Model: ${provider}/${mDir} -> ${join(providerPath, mDir)}`);
          }
        }
      }
    } catch (e) {
      console.log(`Path ${base} not found.`);
    }
  }
}

async function runTest() {
  console.log('=== OS SCAN VERIFICATION (Dry Run) ===');
  await scanOllama();
  await scanLMStudio();
}

runTest();
