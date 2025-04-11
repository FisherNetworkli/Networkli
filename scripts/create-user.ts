import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  const email = process.env.USER_EMAIL || 'user@example.com';
  const password = process.env.USER_PASSWORD || 'user123';

  const hashedPassword = await bcrypt.hash(password, 10);

  try {
    const existingUser = await prisma.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      console.log('User already exists:', email);
      return;
    }

    const user = await prisma.user.create({
      data: {
        email,
        name: 'Regular User',
        password: hashedPassword,
        role: 'USER',
      },
    });

    console.log('Regular user created:', user.email);
  } catch (error) {
    console.error('Error creating user:', error);
  } finally {
    await prisma.$disconnect();
  }
}

main(); 