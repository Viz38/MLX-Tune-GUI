import { NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import { join, relative } from 'path';

export async function GET() {
  const root = join(process.cwd(), '..');
  
  // Folders to exclude or include
  const include = ['mlx_tune', 'data', 'examples', 'outputs', 'tests', 'docs'];
  
  async function getTree(dir: string): Promise<any[]> {
    const entries = await readdir(dir, { withFileTypes: true });
    const items = await Promise.all(
      entries.map(async (entry) => {
        const fullPath = join(dir, entry.name);
        const relPath = relative(root, fullPath);
        
        if (entry.name.startsWith('.') || entry.name === 'node_modules' || entry.name === 'web') {
          return null;
        }

        if (entry.isDirectory()) {
          const children = await getTree(fullPath);
          return {
            id: relPath,
            name: entry.name,
            isDir: true,
            children: children.filter(Boolean),
          };
        } else {
          return {
            id: relPath,
            name: entry.name,
            isDir: false,
          };
        }
      })
    );
    return items.filter(Boolean);
  }

  try {
    const tree = await getTree(root);
    // Filter top level to only include our important directories and root files
    const filteredTree = tree.filter(item => 
      include.includes(item.name) || !item.isDir
    ).sort((a, b) => (b.isDir ? 1 : 0) - (a.isDir ? 1 : 0));
    
    return NextResponse.json(filteredTree);
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
