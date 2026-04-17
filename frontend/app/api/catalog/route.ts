import { NextResponse } from 'next/server';

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? 'http://localhost:8000';

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get('limit') ?? '20';

  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    const controller = new AbortController();
    timeout = setTimeout(() => controller.abort(), 12000);

    const response = await fetch(
      `${BACKEND_API_URL}/catalog?limit=${limit}`,
      { cache: 'no-store', signal: controller.signal }
    );

    const data = await response.json();
    if (!response.ok) {
      return NextResponse.json(
        { detail: data?.detail ?? 'Catalog request failed.' },
        { status: response.status },
      );
    }
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json({ detail: 'Catalog timed out.' }, { status: 504 });
    }
    return NextResponse.json({ detail: 'Unable to reach backend catalog service.' }, { status: 502 });
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
