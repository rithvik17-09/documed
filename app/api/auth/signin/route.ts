import { NextResponse } from 'next/server';
import clientPromise from '@/lib/mongodb';

export async function POST(request: Request) {
  const { email, password } = await request.json();
  const client = await clientPromise;
  const db = client.db();
  const user = await db.collection('users').findOne({ email });

  if (!user || user.password !== password) {
    return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 });
  }

  // You can add JWT/session logic here
  return NextResponse.json({ message: 'Login successful', user: { email: user.email } });
}
