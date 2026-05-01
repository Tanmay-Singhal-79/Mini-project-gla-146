import React from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  ChevronLeft, 
  ChevronRight, 
  Play, 
  Lightbulb, 
  Code2, 
  Home, 
  ArrowLeft, 
  Sparkles, 
  BookOpen, 
  CheckCircle2 
} from 'lucide-react';
import api from '../services/api';
import { Button } from '../components/ui/Button';
import { syncProgress, getRobustProgress } from '../services/progressSync';

import { getFallbackLesson } from '../data/mockLessons';

export const LessonViewer = () => {
  const { title } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  
  const roadmapSteps = location.state?.steps || [];
  
  const { data: lesson, isLoading, isError } = useQuery({
    queryKey: ['lesson', title],
    queryFn: async () => {
      try {
        // Race the API against a 300ms timeout
        const fetchPromise = api.learning.getLesson(title).then(res => res.data);
        const timeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error("HyperTimeout")), 300)
        );

        const data = await Promise.race([fetchPromise, timeoutPromise]);
        
        if (data && data.sections) return data;
        throw new Error("Invalid Data");
      } catch (err) {
        return getFallbackLesson(title);
      }
    },
    enabled: !!title,
  });

  const { data: progress } = useQuery({
    queryKey: ['progress'],
    queryFn: () => getRobustProgress(),
  });

  const { data: profile } = useQuery({
    queryKey: ['profile'],
    queryFn: () => api.auth.me().then(res => res.data),
    enabled: roadmapSteps.length === 0, // Only fetch if we lost state
  });

  const { data: fallbackPath } = useQuery({
    queryKey: ['path', profile?.current_interest],
    queryFn: () => api.learning.generatePath(profile?.current_interest).then(res => res.data.steps),
    enabled: !!profile?.current_interest && roadmapSteps.length === 0,
  });

  // Use state steps if available, otherwise use fallback from API
  const finalSteps = roadmapSteps.length > 0 ? roadmapSteps : (fallbackPath || []);
  const actualCurrentIndex = finalSteps.findIndex(s => s.title === title || s.title === decodeURIComponent(title));

  const completeMutation = useMutation({
    mutationFn: (stepTitle) => syncProgress(stepTitle),
    onMutate: async (stepTitle) => {
      await queryClient.cancelQueries({ queryKey: ['progress'] });
      const previousProgress = queryClient.getQueryData(['progress']);
      queryClient.setQueryData(['progress'], (old) => {
        const completedItems = old?.completedItems || [];
        if (!completedItems.find(item => item.step_title === stepTitle)) {
          const newItems = [...completedItems, { step_title: stepTitle, status: 'completed', id: 'temp_' + Date.now() }];
          const newPercentage = Math.round((newItems.length / 500) * 1000) / 10;
          return {
            ...old,
            completedItems: newItems,
            percentage: newPercentage
          };
        }
        return old;
      });
      return { previousProgress };
    },
    onError: (err, stepTitle, context) => {
      queryClient.setQueryData(['progress'], context.previousProgress);
      console.error("Critical Sync Failure:", err);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });

  const isCompleted = progress?.completedItems?.some(item => item.step_title?.trim() === title?.trim());

  const goToNext = () => {
    if (actualCurrentIndex < finalSteps.length - 1) {
      const nextStep = finalSteps[actualCurrentIndex + 1];
      navigate(`/lesson/${encodeURIComponent(nextStep.title)}`, { state: { steps: finalSteps } });
    }
  };

  const goToPrev = () => {
    if (actualCurrentIndex > 0) {
      const prevStep = finalSteps[actualCurrentIndex - 1];
      navigate(`/lesson/${encodeURIComponent(prevStep.title)}`, { state: { steps: finalSteps } });
    }
  };

  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center">
        <div className="p-8 bg-red-50 text-red-500 rounded-[2rem] shadow-xl">
          <BookOpen className="w-16 h-16" />
        </div>
        <div className="space-y-4">
          <h2 className="text-4xl font-black text-slate-800 tracking-tighter">Transmission Interrupted</h2>
          <p className="text-xl text-slate-500 font-bold max-w-md mx-auto">We couldn't retrieve the curriculum for "{title}". The signal might be weak.</p>
        </div>
        <Button onClick={() => navigate('/learning-path')} className="btn-primary px-10 py-6 rounded-[1.5rem] font-black">
          Return to Neural Map
        </Button>
      </div>
    );
  }

  // Fallback if data is still missing (very rare with our new local cache)
  if (!lesson) return null;

  return (
    <div className="flex min-h-screen bg-[var(--color-background)] -m-8">
      {/* Sidebar - Pro Navigation */}
      <aside className="w-80 glass-card border-r border-primary/10 bg-white/40 backdrop-blur-2xl overflow-y-auto hidden lg:block sticky top-0 h-screen">
        <div className="p-10 border-b border-primary/10">
          <div className="flex items-center gap-4 text-primary font-black text-2xl tracking-tighter">
            <div className="p-3 bg-primary text-white rounded-2xl shadow-lg shadow-primary/20">
              <Sparkles className="w-6 h-6" />
            </div>
            <span>LearnPath</span>
          </div>
        </div>
        <nav className="p-6 space-y-3">
          <p className="px-4 py-2 text-xs font-black text-slate-400 uppercase tracking-[0.2em] mb-4">
            Current Trajectory
          </p>
          {finalSteps.length > 0 ? (
            finalSteps.map((step, idx) => (
              <button
                key={idx}
                onClick={() => navigate(`/lesson/${encodeURIComponent(step.title)}`, { state: { steps: finalSteps } })}
                className={`w-full text-left px-5 py-4 rounded-2xl text-sm font-black transition-all duration-300 ${
                  step.title === title || step.title === decodeURIComponent(title)
                    ? 'bg-primary text-white shadow-xl shadow-primary/20 scale-[1.02]'
                    : 'text-slate-500 hover:bg-primary/5 hover:text-primary'
                }`}
              >
                <span className="opacity-40 mr-2">{(idx + 1).toString().padStart(2, '0')}</span>
                {step.title}
              </button>
            ))
          ) : (
             <button
              onClick={() => navigate('/learning-path')}
              className="w-full text-left px-5 py-4 rounded-2xl text-sm font-black text-primary bg-primary/5 hover:bg-primary/10 flex items-center gap-3 transition-all"
            >
              <Home className="w-5 h-5" /> Return to Map
             </button>
          )}
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto bg-transparent">
        {/* Navigation Header */}
        <header className="sticky top-0 z-20 bg-[var(--color-background)]/80 backdrop-blur-md border-b border-primary/5 px-10 py-6 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Button 
              variant="ghost" 
              onClick={() => navigate('/learning-path')}
              className="lg:hidden text-primary"
            >
              <ArrowLeft className="w-6 h-6" />
            </Button>
            <h2 className="text-xl font-black text-slate-800 tracking-tight truncate max-w-md">
              {lesson.title}
            </h2>
          </div>
          
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost"
              onClick={() => completeMutation.mutate(title)}
              className={`rounded-2xl px-5 py-4 font-black transition-all ${
                isCompleted 
                  ? 'bg-primary/10 text-primary' 
                  : 'text-slate-400 hover:bg-emerald-50 hover:text-primary'
              }`}
            >
              <CheckCircle2 className={`w-6 h-6 ${isCompleted ? 'fill-current' : ''}`} />
            </Button>

            <Button 
              variant="outline" 
              onClick={goToPrev}
              disabled={actualCurrentIndex <= 0}
              className="rounded-2xl px-6 py-4 border-primary/10 text-primary font-black"
            >
              <ChevronLeft className="w-5 h-5" />
            </Button>
            <Button 
              onClick={goToNext}
              disabled={actualCurrentIndex >= finalSteps.length - 1 && finalSteps.length > 0}
              className="btn-primary rounded-2xl px-8 py-4 font-black shadow-lg shadow-primary/20"
            >
              <span className="hidden sm:inline">Continue</span>
              <ChevronRight className="w-5 h-5 ml-1" />
            </Button>
          </div>
        </header>

        {/* Lesson Content */}
        <div className="max-w-5xl mx-auto px-6 md:px-10 py-16 space-y-24">
          {/* Hero Section */}
          <section className="space-y-10 relative">
            <div className="absolute -left-20 top-0 hidden xl:block">
              <div className="sticky top-40 space-y-8">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-1 h-20 bg-primary/10 rounded-full overflow-hidden">
                    <div 
                      className="w-full bg-primary origin-top transition-all duration-700"
                      style={{ height: `${actualCurrentIndex >= 0 ? ((actualCurrentIndex + 1) / (finalSteps.length || 1)) * 100 : 0}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-black text-primary uppercase tracking-widest [writing-mode:vertical-lr] rotate-180">Progress</span>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-emerald-50 text-primary border border-primary/10 shadow-sm">
                <Sparkles className="w-4 h-4 animate-pulse" />
                <span className="text-[10px] font-black uppercase tracking-widest">Active Curriculum Node</span>
              </div>
              <h1 className="text-5xl md:text-8xl font-black text-slate-800 leading-[1.1] tracking-tighter">
                {lesson.title}
              </h1>
              <div 
                className="text-xl md:text-3xl text-slate-500 font-medium leading-relaxed max-w-4xl border-l-4 border-primary/20 pl-8 py-4 bg-primary/5 rounded-r-3xl"
                dangerouslySetInnerHTML={{ __html: lesson.introduction.replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-800">$1</strong>').replace(/`(.*?)`/g, '<code class="bg-primary/10 text-primary px-2 py-0.5 rounded-lg font-black">$1</code>') }}
              />
            </div>
          </section>

          {/* Sections */}
          {lesson?.sections?.map((section, idx) => (
            <section 
              key={idx}
              className="space-y-12"
            >
              <div className="flex items-center gap-8 group">
                <div className="w-20 h-1.5 bg-primary/10 rounded-full overflow-hidden">
                   <div className="w-0 group-hover:w-full h-full bg-primary transition-all duration-700" />
                </div>
                <h2 className="text-3xl md:text-5xl font-black text-slate-800 tracking-tight">
                  {section.heading}
                </h2>
              </div>
              
              {section.content && (
                <div 
                  className="text-lg md:text-2xl text-slate-500 font-medium leading-relaxed max-w-4xl selection:bg-primary/10"
                  dangerouslySetInnerHTML={{ 
                    __html: section.content
                      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-800">$1</strong>')
                      .replace(/`(.*?)`/g, '<code class="bg-primary/5 text-primary px-2 py-1 rounded-lg font-black">$1</code>') 
                  }}
                />
              )}
              
              {section.code && (
                <div className="group relative rounded-[3rem] overflow-hidden border-4 border-white shadow-premium bg-slate-950">
                  <div className="flex items-center justify-between px-10 py-6 bg-slate-900/50 border-b border-white/5">
                    <div className="flex items-center gap-4">
                      <div className="flex gap-2">
                        <div className="w-3.5 h-3.5 rounded-full bg-red-500 shadow-lg shadow-red-500/20" />
                        <div className="w-3.5 h-3.5 rounded-full bg-amber-500 shadow-lg shadow-amber-500/20" />
                        <div className="w-3.5 h-3.5 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/20" />
                      </div>
                      <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.3em] ml-6">
                        {section.language || 'Architecture Fragment'}
                      </span>
                    </div>
                    <Button 
                      variant="ghost" 
                      onClick={() => {
                        navigator.clipboard.writeText(section.code);
                      }}
                      className="text-slate-400 hover:text-white font-black uppercase text-[10px] tracking-widest hover:bg-white/5 px-4 py-2 rounded-xl"
                    >
                      Capture Logic
                    </Button>
                  </div>
                  <pre className="p-12 overflow-x-auto text-lg md:text-xl leading-relaxed text-emerald-400 font-mono scrollbar-hide">
                    <code>{section.code}</code>
                  </pre>
                </div>
              )}
            </section>
          ))}

          {/* Tip Card */}
          {lesson.tip && (
            <div 
              className="glass-card rounded-[4rem] p-16 bg-emerald-50/50 border-primary/20 flex flex-col md:flex-row gap-12 relative overflow-hidden group"
            >
              <div className="absolute -right-20 -bottom-20 w-64 h-64 bg-primary/5 rounded-full blur-3xl group-hover:scale-150 transition-transform duration-1000" />
              <div className="shrink-0 w-24 h-24 bg-white rounded-[2rem] flex items-center justify-center text-primary shadow-2xl shadow-primary/10 relative z-10 group-hover:rotate-12 transition-transform">
                <Lightbulb className="w-12 h-12" />
              </div>
              <div className="space-y-4 relative z-10">
                <h4 className="text-3xl font-black text-primary uppercase tracking-tighter">Pro Insight</h4>
                <p className="text-xl md:text-2xl text-slate-600 font-bold leading-relaxed">
                  {lesson.tip}
                </p>
              </div>
            </div>
          )}

          {/* Bottom Navigation */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-12 pt-24 border-t-2 border-primary/10">
            <Button 
              variant="outline" 
              onClick={goToPrev}
              disabled={actualCurrentIndex <= 0}
              className="w-full md:w-auto rounded-[2rem] px-12 py-10 text-xl font-black border-primary/20 text-primary hover:bg-primary/5 h-auto"
            >
              <ChevronLeft className="w-8 h-8 mr-4" />
              Previous
            </Button>
            
            <div className="flex flex-col items-center gap-2">
              <span className="text-slate-400 font-black uppercase tracking-[0.3em] text-[10px]">Neural Sequence</span>
              <div className="text-4xl font-black text-slate-800">
                {actualCurrentIndex >= 0 ? actualCurrentIndex + 1 : 0} <span className="text-slate-300 mx-2">/</span> {finalSteps.length}
              </div>
            </div>

            <Button 
              onClick={() => {
                import('canvas-confetti').then(confetti => {
                    confetti.default({
                        particleCount: 150,
                        spread: 70,
                        origin: { y: 0.6 },
                        colors: ['#00A19B', '#34d399', '#ffffff']
                    });
                });
                completeMutation.mutate(title);
                setTimeout(goToNext, 1500);
              }}
              disabled={actualCurrentIndex >= finalSteps.length - 1 && finalSteps.length > 0}
              className="w-full md:w-auto btn-primary rounded-[2rem] px-16 py-10 text-2xl font-black shadow-2xl shadow-primary/30 h-auto"
            >
              Complete Milestone
              <ChevronRight className="w-8 h-8 ml-4" />
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
};
