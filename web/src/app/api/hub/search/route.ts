import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get('q');

  if (!q) {
    return NextResponse.json([]);
  }

  try {
    // Query Hugging Face Model API
    // We filter by 'mlx' tag to ensure compatibility
    const url = `https://huggingface.co/api/models?search=${encodeURIComponent(q)}&filter=mlx&sort=downloads&direction=-1&limit=20`;
    
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('HF API responded with an error');
    }

    const data = await response.json();
    
    // Map to a cleaner format for the GUI
    const models = data.map((m: any) => ({
      id: m.modelId,
      name: m.modelId,
      downloads: m.downloads,
      likes: m.likes,
      updated: m.lastModified,
      author: m.author,
      isMLX: true
    }));

    return NextResponse.json(models);
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
