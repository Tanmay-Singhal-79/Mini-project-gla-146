import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export const Button = React.forwardRef(({ className, variant = 'primary', ...props }, ref) => {
  const baseStyles = "inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none ring-offset-white";
  const variants = {
    primary: "bg-primary text-white hover:bg-primary-dark focus-visible:ring-primary shadow-lg shadow-primary/20",
    secondary: "bg-emerald-50 text-primary hover:bg-emerald-100 focus-visible:ring-primary",
    outline: "border-2 border-primary/20 hover:border-primary/40 text-primary hover:bg-primary/5 focus-visible:ring-primary",
    ghost: "hover:bg-primary/5 hover:text-primary text-slate-500",
  };

  return (
    <button
      ref={ref}
      className={cn(baseStyles, variants[variant], "h-10 py-2 px-4", className)}
      {...props}
    />
  );
});
Button.displayName = "Button";
