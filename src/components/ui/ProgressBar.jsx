import React from 'react';
import { cn } from './Button';

export const ProgressBar = ({ value, className, ...props }) => {
  return (
    <div
      className={cn("relative h-4 w-full overflow-hidden rounded-full bg-slate-100", className)}
      {...props}
    >
      <div
        className="h-full w-full flex-1 bg-blue-600 transition-all duration-500 ease-in-out"
        style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
      />
    </div>
  );
};
