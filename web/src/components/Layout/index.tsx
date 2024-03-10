// components/Layout.tsx
import Head from 'next/head';
import React, { ReactNode } from 'react';

import { Navbar } from '@/components/NavBar';
import { SubNavigationBar } from '@/components/SubNavigationBar';
import { SubNavigationMenu } from '@/components/shared/SubNavigationMenu';
import { Footer } from '@/components/Footer';
import styles from '@/styles/Index.module.scss';

type Props = {
  children: ReactNode;
  localNav?: ReactNode;
  pageTitle?: string; // Optional prop for setting the page title
};

const Layout: React.FC<Props> = ({ children, localNav, pageTitle }) => {
  return (
    <div className={styles.container}>
      <Head>
        {/* Set a dynamic title if provided */}
        {pageTitle && <title>{pageTitle} | SciPhi</title>}
        {/* You can also include other page-specific meta tags, links, or scripts here */}
      </Head>
      <Navbar />
      {/* <SubNavigationBar /> */}
      {/* <SubNavigationMenu /> */}
      <main className={styles.main}>{children}</main>
      <Footer />
    </div>
  );
};

export default Layout;
