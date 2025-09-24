import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// Reset password using OTP
export async function PATCH(req: Request) {
  const { gmail, otp, newPassword } = await req.json();
  const user = await prisma.user.findUnique({ where: { gmail } });
  if (!user) {
    return NextResponse.json({ error: "Invalid OTP or user" }, { status: 400 });
  }
  const tokenRecord = await prisma.passwordResetToken.findFirst({
    where: { otp, userId: user.id },
  });
  if (!tokenRecord || new Date(tokenRecord.expiresAt) < new Date()) {
    return NextResponse.json({ error: "Invalid or expired OTP" }, { status: 400 });
  }
  await prisma.user.update({ where: { id: user.id }, data: { password: newPassword } });
  await prisma.passwordResetToken.delete({ where: { id: tokenRecord.id } });
  return NextResponse.json({ success: true });
}
