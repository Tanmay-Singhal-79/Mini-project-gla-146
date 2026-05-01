import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Map, 
  Compass, 
  TrendingUp, 
  Users, 
  Code 
} from 'lucide-react';
import { cn } from '../ui/Button';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Path', href: '/learning-path', icon: Map },
  { name: 'Discover', href: '/recommendations', icon: Compass },
  { name: 'Progress', href: '/progress', icon: TrendingUp },
  { name: 'Community', href: '/community', icon: Users },
];

export const MobileNav = () => {
  return (
    <nav className="lg:hidden fixed bottom-6 left-6 right-6 z-50 bg-white/80 backdrop-blur-2xl border border-primary/10 rounded-[2.5rem] px-4 py-3 shadow-2xl shadow-primary/20">
      <div className="flex items-center justify-between">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              cn(
                'flex flex-col items-center gap-1 p-3 rounded-2xl transition-all duration-300',
                isActive 
                  ? 'bg-primary text-white scale-110 shadow-lg shadow-primary/20' 
                  : 'text-slate-400'
              )
            }
          >
            <item.icon className="w-5 h-5" />
            <span className="text-[10px] font-black uppercase tracking-widest hidden sm:block">
              {item.name}
            </span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
};
