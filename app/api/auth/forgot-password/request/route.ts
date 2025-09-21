import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// Request password reset (send token)
export async function PUT(req: Request) {
  const { username } = await req.json();
  const user = await prisma.user.findUnique({ where: { username } });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }
  const resetToken = Math.random().toString(36).substring(2, 10);
  await prisma.passwordResetToken.create({ data: { token: resetToken, userId: user.id } });
  // In a real app, email this token to the user
  return NextResponse.json({ success: true, resetToken });
}
