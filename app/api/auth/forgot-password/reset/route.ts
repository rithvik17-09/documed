import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

// Reset password using token
export async function PATCH(req: Request) {
  const { username, resetToken, newPassword } = await req.json();
  const user = await prisma.user.findUnique({ where: { username } });
  if (!user) {
    return NextResponse.json({ error: "Invalid token or user" }, { status: 400 });
  }
  const tokenRecord = await prisma.passwordResetToken.findFirst({
    where: { token: resetToken, userId: user.id },
  });
  if (!tokenRecord) {
    return NextResponse.json({ error: "Invalid token or user" }, { status: 400 });
  }
  await prisma.user.update({ where: { id: user.id }, data: { password: newPassword } });
  await prisma.passwordResetToken.delete({ where: { id: tokenRecord.id } });
  return NextResponse.json({ success: true });
}
