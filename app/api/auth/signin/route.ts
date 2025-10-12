import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request: Request) {
  const { email, password } = await request.json();
  
  const user = await prisma.user.findFirst({
    where: {
      OR: [
        { gmail: email },
        { username: email }
      ]
    }
  });

  if (!user || user.password !== password) {
    return NextResponse.json({ error: 'Invalid credentials' }, { status: 401 });
  }
  return NextResponse.json({ 
    message: 'Login successful', 
    user: { 
      id: user.id,
      username: user.username,
      email: user.gmail 
    } 
  });
}
