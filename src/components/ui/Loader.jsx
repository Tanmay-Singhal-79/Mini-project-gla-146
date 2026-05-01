import React from 'react';
import { Loader2 } from 'lucide-react';
import { cn } from './Button';

export const Loader = ({ className, size = 24, ...props }) => {
  return (
    <Loader2 
      size={size} 
      className={cn("animate-spin text-blue-600", className)} 
      {...props} 
    />
  );
};

export const FullPageLoader = () => (
  <div className="flex h-screen w-full items-center justify-center bg-slate-50/50">
    <Loader size={48} />
  </div>
);
