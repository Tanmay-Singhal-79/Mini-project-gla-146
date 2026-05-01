import React, { useState, useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { Navbar } from './Navbar';
import { MobileNav } from './MobileNav';
import { PageWrapper } from './PageWrapper';

export const AppLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth > 1024);
  const location = useLocation();

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 1024) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="flex h-screen bg-[var(--color-background)] selection:bg-primary/20 selection:text-primary overflow-hidden relative">
      {/* Desktop Sidebar */}
      <div className={`hidden lg:block transition-all duration-500 ease-in-out ${sidebarOpen ? 'w-72' : 'w-0'} h-full overflow-hidden shrink-0`}>
        <Sidebar isOpen={sidebarOpen} />
      </div>

      <div className="flex flex-1 flex-col overflow-hidden min-w-0 relative">
        <Navbar toggleSidebar={() => setSidebarOpen(!sidebarOpen)} sidebarOpen={sidebarOpen} />
        
        <main className="flex-1 overflow-y-auto p-4 md:p-12 custom-scrollbar pb-32 lg:pb-12">
          <div className="mx-auto max-w-7xl">
            <AnimatePresence mode="wait">
              <PageWrapper key={location.pathname}>
                <Outlet />
              </PageWrapper>
            </AnimatePresence>
          </div>
        </main>

        <MobileNav />
      </div>
    </div>
  );
};
