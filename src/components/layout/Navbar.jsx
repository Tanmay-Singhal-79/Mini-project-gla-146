import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, Bell, User, Menu, X, Search, Sparkles } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { Button } from '../ui/Button';

export const Navbar = ({ toggleSidebar, sidebarOpen }) => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate('/learning-path', { state: { interest: searchQuery.trim() } });
      setSearchQuery('');
    }
  };

  return (
    <header className="flex h-24 shrink-0 items-center justify-between px-10 bg-transparent sticky top-0 z-30 transition-all duration-300">
      <div className="flex items-center gap-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="text-primary hover:bg-primary/5 rounded-xl p-2.5 transition-all duration-300"
        >
          {sidebarOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </Button>
        
        <form 
          onSubmit={handleSearch}
          className="hidden md:flex items-center gap-3 bg-white/50 border border-primary/10 rounded-2xl px-6 py-2.5 focus-within:ring-2 focus-within:ring-primary/20 transition-all duration-300 group"
        >
          <Search className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors cursor-pointer" onClick={handleSearch} />
          <input 
            type="text" 
            placeholder="Search learning topics..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-transparent border-none outline-none text-sm font-bold text-slate-600 placeholder:text-slate-400 w-64"
          />
        </form>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-2xl bg-emerald-50 text-primary border border-primary/10 animate-pulse">
          <Sparkles className="h-4 w-4" />
          <span className="text-[10px] font-black uppercase tracking-widest">Neural Link Active</span>
        </div>

        <button 
          type="button" 
          onClick={() => alert('No new notifications at this time.')}
          className="relative p-2.5 text-slate-400 hover:text-primary hover:bg-primary/5 rounded-xl transition-all duration-300"
        >
          <Bell className="h-6 w-6" />
          <span className="absolute top-2 right-2 h-2.5 w-2.5 rounded-full bg-primary border-2 border-[var(--color-background)]" />
        </button>

        <div className="h-8 w-px bg-primary/10 mx-2" />

        <div className="flex items-center gap-4">
          <div className="flex flex-col items-end hidden lg:flex">
            <span className="text-sm font-black text-slate-800 leading-tight">
              {user?.displayName || 'Developer'}
            </span>
            <span className="text-[10px] font-black text-primary uppercase tracking-widest">
              Lvl 12 Architect
            </span>
          </div>
          <div 
            onClick={() => navigate('/progress')}
            className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary text-white shadow-lg shadow-primary/20 hover:scale-110 transition-transform duration-500 cursor-pointer"
            title="View Progress"
          >
            <User className="h-6 w-6" />
          </div>
          <Button 
            variant="ghost" 
            className="p-2.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all" 
            onClick={logout}
          >
            <LogOut className="h-6 w-6" />
          </Button>
        </div>
      </div>
    </header>
  );
};
