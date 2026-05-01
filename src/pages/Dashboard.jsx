import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { 
  Rocket, 
  Target, 
  Sparkles, 
  ArrowRight, 
  Zap,
  TrendingUp,
  Brain,
  Trophy,
  Users,
  Star,
  PlayCircle
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';
import { Button } from '../components/ui/Button';

export const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();

  // 1. Fetch Profile
  const { data: profile, isLoading: profileLoading } = useQuery({
    queryKey: ['profile'],
    queryFn: () => api.auth.me().then(res => res.data),
  });

  // 2. Fetch Progress
  const { data: progress, isLoading: progressLoading } = useQuery({
    queryKey: ['progress'],
    queryFn: () => api.progress.get().then(res => res.data),
  });

  // 3. Fetch Path (to show next steps)
  const interest = profile?.current_interest;
  const { data: pathSteps } = useQuery({
    queryKey: ['path', interest],
    queryFn: () => api.learning.generatePath(interest).then(res => res.data.steps),
    enabled: !!interest,
  });

  // 4. Fetch Favorites
  const { data: favorites } = useQuery({
    queryKey: ['favorites'],
    queryFn: () => api.progress.getFavorites().then(res => res.data),
  });

  const isLoading = profileLoading || progressLoading;

  // Find next 2 non-completed steps
  const completedTitles = new Set(progress?.completedItems?.map(item => item.step_title) || []);
  const curationSteps = pathSteps?.filter(step => !completedTitles.has(step.title)).slice(0, 2) || [];

  return (
    <div className="space-y-12 pb-12">
      {/* Hero Header */}
      <section className="relative overflow-hidden rounded-[3.5rem] bg-primary p-16 text-white shadow-[0_20px_50px_rgba(0,161,155,0.3)]">
        <div className="absolute top-0 right-0 -translate-y-1/4 translate-x-1/4 opacity-10">
          <Sparkles className="h-96 w-96 rotate-12" />
        </div>
        
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/10 backdrop-blur-xl border border-white/20 mb-8">
            <Zap className="w-4 h-4 text-emerald-300" />
            <span className="text-xs font-black tracking-widest uppercase">
              Current Focus: {profile?.current_interest || 'Choose a Domain'}
            </span>
          </div>
          
          <h1 className="text-6xl font-black mb-6 tracking-tight leading-[1.1]">
            Elevate your <br/>
            <span className="text-emerald-300">Potential, {user?.displayName?.split(' ')[0] || 'Developer'}</span>
          </h1>
          
          <p className="text-xl text-emerald-50/80 mb-10 font-medium leading-relaxed max-w-xl">
            {profile?.current_interest 
              ? `Your personalized learning trajectory for ${profile.current_interest} is ready. Tackling your next milestone will put you in the top 1% of learners.`
              : 'Start your journey by selecting a domain you want to master. Our AI will architect a path for you.'}
          </p>
          
          <div className="flex flex-wrap gap-5">
            <Button 
              onClick={() => navigate(profile?.current_interest ? '/learning-path' : '/recommendations')}
              className="bg-white text-primary hover:bg-emerald-50 px-10 py-8 rounded-[2rem] text-lg font-black shadow-2xl transition-all hover:scale-105 active:scale-95"
            >
              {profile?.current_interest ? 'Resume Path' : 'Start Path'}
            </Button>
            <Button 
              variant="outline"
              onClick={() => navigate('/progress')}
              className="border-white/30 text-white hover:bg-white/10 px-10 py-8 rounded-[2rem] text-lg font-black backdrop-blur-md transition-all"
            >
              View Stats
            </Button>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        {/* Main Content - Curation */}
        <div className="lg:col-span-2 space-y-10">
          <div className="flex items-center justify-between">
            <h2 className="text-4xl font-black text-slate-800 tracking-tight">Next Milestones</h2>
            <Button variant="ghost" onClick={() => navigate('/learning-path')} className="text-primary font-black hover:bg-primary/5 rounded-2xl px-6">
              View Roadmap <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {isLoading ? (
              [1, 2].map(i => (
                <div key={i} className="h-72 glass-card rounded-[3rem] animate-pulse" />
              ))
            ) : curationSteps.length > 0 ? (
              curationSteps.map((step, idx) => (
                <div 
                  key={idx}
                  onClick={() => navigate(`/lesson/${encodeURIComponent(step.title)}`)}
                  className="glass-card group p-10 rounded-[3rem] hover:shadow-[0_30px_60px_-15px_rgba(0,161,155,0.15)] hover:scale-[1.02] transition-all duration-500 cursor-pointer relative overflow-hidden flex flex-col h-full"
                >
                  <div className={`absolute top-0 right-0 w-32 h-32 ${idx === 0 ? 'bg-primary/5' : 'bg-emerald-500/5'} rounded-bl-[4rem] -mr-10 -mt-10 group-hover:bg-primary/10 transition-colors`} />
                  <div className="w-16 h-16 bg-emerald-50 rounded-[1.5rem] flex items-center justify-center mb-8 group-hover:scale-110 group-hover:rotate-6 transition-transform duration-500">
                    <Brain className="text-primary w-8 h-8" />
                  </div>
                  <h3 className="text-2xl font-black mb-4 text-slate-800 line-clamp-2">{step.title}</h3>
                  <p className="text-slate-500 font-bold mb-8 leading-relaxed line-clamp-2">
                    {step.description || "Deep dive into the core principles and architectural foundations of this concept."}
                  </p>
                  <div className="flex items-center justify-between mt-auto">
                    <span className="mint-badge font-black px-4 py-2 uppercase text-[10px]">Step {pathSteps?.indexOf(step) + 1}</span>
                    <div className="w-12 h-12 rounded-2xl bg-primary/5 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white transition-all duration-500">
                      <ArrowRight className="w-6 h-6" />
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="md:col-span-2 text-center py-20 glass-card rounded-[3rem] border-dashed border-2 border-primary/10 bg-white/20">
                <p className="text-2xl font-black text-slate-400">
                  {interest ? "You've completed all steps in this path!" : "No path active yet."}
                </p>
                <Button 
                  onClick={() => navigate('/recommendations')}
                  variant="ghost" 
                  className="mt-4 text-primary font-black"
                >
                  Choose New Domain
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Analytics */}
        <div className="space-y-10">
          <h2 className="text-4xl font-black text-slate-800 tracking-tight">Analytics</h2>
          <div className="glass-card p-10 rounded-[3.5rem] relative overflow-hidden">
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-10">
                <div className="w-16 h-16 bg-primary text-white rounded-[1.5rem] flex items-center justify-center shadow-2xl shadow-primary/30">
                  <Trophy className="w-8 h-8" />
                </div>
                <div className="text-right">
                  <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-1">XP Points</p>
                  <p className="text-3xl font-black text-primary">{(progress?.completedItems?.length || 0) * 500}</p>
                </div>
              </div>
              
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between items-end mb-4">
                    <span className="font-black text-slate-700 text-lg">Overall Mastery</span>
                    <span className="font-black text-primary text-2xl">{progress?.percentage || 0}%</span>
                  </div>
                  <div className="h-4 w-full bg-emerald-100 rounded-full overflow-hidden p-1">
                    <div 
                      className="h-full bg-primary rounded-full shadow-[0_0_20px_rgba(0,161,155,0.5)] transition-all duration-1000" 
                      style={{ width: `${progress?.percentage || 0}%` }}
                    />
                  </div>
                </div>
                
                <div className="pt-10 border-t border-slate-100/50">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-emerald-50/50 p-6 rounded-[2rem] border border-primary/5">
                      <p className="text-[10px] font-black text-primary/60 uppercase mb-2">Starred</p>
                      <p className="text-2xl font-black text-slate-800">{favorites?.length || 0}</p>
                    </div>
                    <div className="bg-emerald-50/50 p-6 rounded-[2rem] border border-primary/5">
                      <p className="text-[10px] font-black text-primary/60 uppercase mb-2">Done</p>
                      <p className="text-2xl font-black text-slate-800">{progress?.completedItems?.length || 0}</p>
                    </div>
                  </div>
                </div>

                <Button 
                  onClick={() => navigate('/progress')}
                  className="w-full bg-primary/5 text-primary hover:bg-primary hover:text-white py-6 rounded-[1.5rem] font-black transition-all duration-500"
                >
                  View Full Report
                </Button>
              </div>
            </div>
          </div>

          <div className="glass-card p-8 rounded-[2.5rem] bg-white/40 border-dashed border-2 border-primary/10">
            <div className="flex items-center gap-5">
              <div className="flex -space-x-4">
                {[1,2,3].map(i => (
                  <div key={i} className="w-10 h-10 rounded-full border-2 border-white bg-primary/20 flex items-center justify-center">
                    <Users className="w-5 h-5 text-primary/40" />
                  </div>
                ))}
              </div>
              <p className="text-sm font-bold text-slate-500">
                <span className="text-primary">Community</span> active in {interest || 'AI'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
