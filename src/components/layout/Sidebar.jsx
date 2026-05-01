import React from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Map, 
  Compass, 
  TrendingUp, 
  Users, 
  Code, 
  LogOut, 
  User,
  GraduationCap,
  Sparkles
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../ui/Button';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Learning Path', href: '/learning-path', icon: Map },
  { name: 'Recommendations', href: '/recommendations', icon: Compass },
  { name: 'Progress', href: '/progress', icon: TrendingUp },
  { name: 'Community', href: '/community', icon: Users },
  { name: 'Code Editor', href: '/editor', icon: Code },
];

export const Sidebar = ({ isOpen }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('Failed to log out', error);
    }
  };

  return (
    <div className={`flex h-full w-72 flex-col bg-white/40 backdrop-blur-2xl border-r border-primary/10 transition-all duration-500 ease-in-out ${!isOpen && '-translate-x-full'}`}>
      {/* Logo Section */}
      <div className="flex h-24 shrink-0 items-center px-8">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-white shadow-lg shadow-primary/20 group-hover:scale-110 group-hover:rotate-6 transition-all duration-500">
            <GraduationCap className="h-7 w-7" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-black tracking-tight text-primary leading-tight">LearnPath</span>
            <span className="text-[10px] font-black tracking-[0.2em] text-slate-400 uppercase leading-none">Intelligence</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex flex-1 flex-col overflow-y-auto px-4 py-8">
        <nav className="flex-1 space-y-3">
          <p className="px-5 py-2 text-[10px] font-black text-slate-400 uppercase tracking-[0.3em] mb-4">
            Navigation Hub
          </p>
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'group flex items-center rounded-2xl px-5 py-4 text-sm font-black transition-all duration-500',
                  isActive 
                    ? 'bg-primary text-white shadow-2xl shadow-primary/30 scale-[1.02]' 
                    : 'text-slate-500 hover:bg-primary/5 hover:text-primary'
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon
                    className={cn(
                      'mr-4 h-5 w-5 shrink-0 transition-transform duration-500',
                      isActive ? 'scale-110' : 'group-hover:scale-110'
                    )}
                    aria-hidden="true"
                  />
                  <span className="flex-1 whitespace-nowrap">{item.name}</span>
                  {isActive && (
                    <div className="h-2 w-2 rounded-full bg-white animate-pulse" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Upgrade Card */}
        <div className="mt-8 px-2">
          <div className="bg-primary/5 rounded-[2rem] p-6 border border-primary/10 relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-16 h-16 bg-primary/10 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-700" />
            <Sparkles className="w-8 h-8 text-primary mb-4" />
            <p className="text-sm font-black text-slate-800 mb-2">Upgrade Neural Capacity</p>
            <p className="text-xs text-slate-500 font-bold mb-4">Access premium ML architectures.</p>
            <button className="w-full bg-white text-primary text-[10px] font-black uppercase tracking-widest py-3 rounded-xl shadow-sm hover:shadow-md transition-all">
              Go Pro »
            </button>
          </div>
        </div>
      </div>

      {/* User Section */}
      <div className="p-6 border-t border-primary/10 bg-white/20">
        <div className="bg-white/50 rounded-3xl p-5 border border-primary/5 group hover:border-primary/20 transition-all duration-500">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-primary text-white rounded-2xl flex items-center justify-center shadow-lg shadow-primary/20 group-hover:scale-110 transition-transform duration-500">
              <User className="w-6 h-6" />
            </div>
            <div className="overflow-hidden">
              <p className="text-sm font-black text-slate-800 truncate">
                {user?.displayName || 'Developer'}
              </p>
              <p className="text-[10px] font-black text-primary uppercase tracking-widest truncate">
                Active Node
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
