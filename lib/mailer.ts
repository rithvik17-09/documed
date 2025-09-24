import nodemailer from 'nodemailer';


const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.GMAIL_USER,
    pass: process.env.GMAIL_PASS,
  },
});

export async function sendSignupMail(to: string) {
  const mailOptions = {
    from: process.env.GMAIL_USER,
    to,
    subject: 'Signup Successful',
    text: 'Welcome! Your signup was successful.',
  };
  await transporter.sendMail(mailOptions);
}

export async function sendOtpMail(to: string, otp: string) {
  const mailOptions = {
    from: process.env.GMAIL_USER,
    to,
    subject: 'Password Reset OTP',
    text: `Your OTP for password reset is: ${otp}. It is valid for 10 minutes.`,
  };
  await transporter.sendMail(mailOptions);
}
