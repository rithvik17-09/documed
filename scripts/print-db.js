const { PrismaClient } = require('@prisma/client');

async function main() {
  const prisma = new PrismaClient();
  try {
    const users = await prisma.user.findMany();
    const tokens = await prisma.passwordResetToken.findMany();
    console.log('USERS:', JSON.stringify(users, null, 2));
    console.log('PASSWORD_RESET_TOKENS:', JSON.stringify(tokens, null, 2));
  } catch (e) {
    console.error('Error querying DB:', e);
    process.exitCode = 2;
  } finally {
    await prisma.$disconnect();
  }
}

main();
