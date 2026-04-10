import { NextResponse } from 'next/server';
import { readFile, writeFile } from 'fs/promises';
import { join } from 'path';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const path = searchParams.get('path');

  if (!path) {
    return NextResponse.json({ error: 'Path is required' }, { status: 400 });
  }

  try {
    const root = join(process.cwd(), '..');
    const fullPath = join(root, path);

    // Security: Ensure path is within workspace root
    if (!fullPath.startsWith(root)) {
      return NextResponse.json({ error: 'Unauthorized path' }, { status: 403 });
    }

    const content = await readFile(fullPath, 'utf-8');
    return NextResponse.json({ content });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const { path, content } = await request.json();
    
    if (!path || content === undefined) {
      return NextResponse.json({ error: 'Path and content are required' }, { status: 400 });
    }

    const root = join(process.cwd(), '..');
    const fullPath = join(root, path);

    // Security: Ensure path is within workspace root
    if (!fullPath.startsWith(root)) {
      return NextResponse.json({ error: 'Unauthorized path' }, { status: 403 });
    }

    await writeFile(fullPath, content, 'utf-8');
    return NextResponse.json({ success: true });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
