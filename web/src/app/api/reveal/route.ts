import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import { stat } from 'fs/promises';

const execAsync = promisify(exec);

export async function POST(request: Request) {
  try {
    const { path } = await request.json();
    
    if (!path) {
      return NextResponse.json({ error: 'Path is required' }, { status: 400 });
    }

    // Security: Check if path exists first
    try {
      await stat(path);
    } catch (e) {
      return NextResponse.json({ error: 'Path does not exist' }, { status: 404 });
    }

    // On macOS, 'open' opens the folder in Finder
    await execAsync(`open "${path}"`);

    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
