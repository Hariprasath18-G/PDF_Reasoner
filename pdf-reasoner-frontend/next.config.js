/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    webpack(config) {
      config.module.rules.push({
        test: /\.pdf$/,
        use: 'file-loader',
      });
      return config;
    },
  };
  
  module.exports = nextConfig;