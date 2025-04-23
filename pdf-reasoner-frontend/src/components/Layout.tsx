import Head from 'next/head';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <>
    <Head>
      <title>PDF Reasoner</title>
      <meta name="description" content="Interactive PDF Reasoner with AI agents" />
    </Head>
    {children}
  </>
);

export default Layout;