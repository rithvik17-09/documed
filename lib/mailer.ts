import nodemailer from 'nodemailer';


const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.GMAIL_USER,
    pass: process.env.GMAIL_PASS,
  },
});

export async function sendSignupMail(to: string, name: string) {
  const mailOptions = {
    from: process.env.GMAIL_USER,
    to,
    subject: 'Welcome to DocuMed! 🎉',
    text: `Dear ${name},\n\nWe’re excited to welcome you to DocuMed! 🎉\nThank you for joining our community dedicated to making healthcare smarter, faster, and more accessible.\n\nWith DocuMed, you can:\n\n-> Manage and analyze medical documents easily\n-> Get AI-powered insights and summaries\n\nWe’re thrilled to have you onboard and can’t wait for you to experience everything DocuMed has to offer.\n\nIf you have any questions, feel free to reach out to us at ortest1990@gmail.com\n\nOnce again, welcome to the DocuMed family! 💙\n\nWarm regards,\nTeam DocuMed`,
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
