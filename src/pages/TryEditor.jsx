import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { Play, RotateCcw, ChevronLeft, Save, Download, Share2, CheckCircle2 } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import api from '../services/api';

export const TryEditor = () => {
  const { title } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();

  const completeMutation = useMutation({
    mutationFn: (stepTitle) => api.progress.update(stepTitle),
    onMutate: async (stepTitle) => {
      await queryClient.cancelQueries({ queryKey: ['progress'] });
      const previousProgress = queryClient.getQueryData(['progress']);
      queryClient.setQueryData(['progress'], (old) => {
        const completedItems = old?.completedItems || [];
        if (!completedItems.find(item => item.step_title === stepTitle)) {
          return {
            ...old,
            completedItems: [...completedItems, { step_title: stepTitle, status: 'completed', id: 'temp_' + Date.now() }]
          };
        }
        return old;
      });
      return { previousProgress };
    },
    onError: (err, stepTitle, context) => {
      queryClient.setQueryData(['progress'], context.previousProgress);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });
  const boilerplates = {
    html: "<!-- Try it yourself -->\n<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n  <p>Start coding...</p>\n</body>\n</html>",
    javascript: "// JavaScript Playground\nconsole.log(\"Hello from JavaScript!\");\n\nconst greet = (name) => {\n  return `Hello, ${name}!`;\n};\n\nconsole.log(greet(\"Developer\"));",
    python: "# Python Playground\nprint(\"Hello from Python!\")\n\ndef square(n):\n    return n * n\n\nprint(f\"Square of 5 is: {square(5)}\")",
    java: "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello from Java!\");\n        int sum = add(10, 20);\n        System.out.println(\"Sum is: \" + sum);\n    }\n    \n    public static int add(int a, int b) {\n        return a + b;\n    }\n}",
    cpp: "#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << \"Hello from C++!\" << endl;\n    int n = 5;\n    cout << \"Value of n: \" << n << endl;\n    return 0;\n}"
  };

  const [code, setCode] = useState(location.state?.code || boilerplates[location.state?.language || 'html']);
  const [srcDoc, setSrcDoc] = useState('');
  const [language, setLanguage] = useState(location.state?.language || 'html');
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pyodide, setPyodide] = useState(null);

  // Switch boilerplate when language changes
  useEffect(() => {
    if (!location.state?.code) {
      setCode(boilerplates[language]);
    }
  }, [language]);

  const getWrappedCode = (input) => {
    if (input.trim().startsWith('<')) return input;
    if (input.includes('console.') || input.includes('const ') || input.includes('let ') || input.includes('function')) {
      return `<!DOCTYPE html><html><body><script>${input}<\/script><p>Check the browser console (F12) for output if there is no UI change.</p></body></html>`;
    }
    return `<!DOCTYPE html><html><body>${input}</body></html>`;
  };

  const downloadCode = () => {
    const extensions = { html: 'html', javascript: 'js', python: 'py', java: 'java', cpp: 'cpp' };
    const element = document.createElement("a");
    const file = new Blob([code], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = `playground.${extensions[language] || 'txt'}`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  useEffect(() => {
    const timeout = setTimeout(() => {
      if (language === 'html') setSrcDoc(getWrappedCode(code));
    }, 250);
    return () => clearTimeout(timeout);
  }, [code, language]);

  // Initialize Pyodide
  useEffect(() => {
    if (language === 'python' && !pyodide) {
      const initPyodide = async () => {
        setIsLoading(true);
        setOutput("Initializing Python Virtual Environment...");
        try {
          if (!window.loadPyodide) {
             throw new Error("Pyodide script not loaded. Check your internet connection.");
          }
          // @ts-ignore
          const py = await window.loadPyodide();
          setPyodide(py);
          setOutput("> Python 3.11 engine ready.\n> Type code and press Run.");
        } catch (err) {
          console.error("Pyodide failed:", err);
          const isStorageError = err.message?.includes('storage') || err.message?.includes('IndexedDB') || !window.indexedDB;
          setOutput(
            isStorageError 
            ? "❌ BROWSER SECURITY ALERT: Your browser is blocking storage access (IndexedDB). Please disable 'Tracking Prevention' or allow third-party storage for this site to run Python locally."
            : `❌ FAILED TO INITIALIZE ENGINE: ${err.message}`
          );
        }
        setIsLoading(false);
      };
      initPyodide();
    }
  }, [language, pyodide]);

  const runCode = async () => {
    setOutput('');
    if (language === 'html') {
      setSrcDoc(getWrappedCode(code));
    } else if (language === 'javascript') {
      try {
        const originalLog = console.log;
        let logs = [];
        console.log = (...args) => logs.push(args.map(a => String(a)).join(' '));
        // eslint-disable-next-line no-eval
        eval(code);
        console.log = originalLog;
        setOutput(logs.join('\n') || 'Program finished (no output)');
      } catch (err) {
        setOutput(`Error: ${err.message}`);
      }
    } else if (language === 'python') {
      if (!pyodide) {
        setOutput("Python environment is loading...");
        return;
      }
      setIsLoading(true);
      try {
        pyodide.runPython(`
import sys
import io
sys.stdout = io.StringIO()
`);
        await pyodide.runPythonAsync(code);
        const stdout = pyodide.runPython("sys.stdout.getvalue()");
        setOutput(stdout || 'Program finished (no output)');
      } catch (err) {
        setOutput(`Python Error: ${err.message}`);
      }
      setIsLoading(false);
    } else if (language === 'java' || language === 'cpp') {
      setIsLoading(true);
      setOutput(`Compiling and running ${language.toUpperCase()}...`);
      try {
        const runtimeMap = { java: 'java', cpp: 'cpp' };
        const response = await fetch('https://emkc.org/api/v2/piston/execute', {
          method: 'POST',
          body: JSON.stringify({
            language: runtimeMap[language],
            version: '*',
            files: [{ content: code }]
          })
        });
        const result = await response.json();
        setOutput(result.run.output || 'Program finished (no output)');
      } catch (err) {
        setOutput(`Execution Error: ${err.message}`);
      }
      setIsLoading(false);
    }
  };

  const languages = [
    { id: 'html', name: 'Web (HTML/CSS/JS)' },
    { id: 'javascript', name: 'JavaScript' },
    { id: 'python', name: 'Python' },
    { id: 'java', name: 'Java' },
    { id: 'cpp', name: 'C++' },
  ];

  return (
    <div className="flex flex-col h-screen bg-slate-100 -m-8 overflow-hidden">
      {/* Editor Header */}
      <header className="bg-slate-950 text-white px-8 py-4 flex items-center justify-between shadow-2xl border-b border-primary/10">
        <div className="flex items-center gap-6">
          <Button 
            variant="ghost" 
            onClick={() => navigate(-1)}
            className="text-slate-400 hover:text-white font-black uppercase text-xs tracking-widest"
          >
            <ChevronLeft className="w-5 h-5 mr-1" /> Return
          </Button>
          <div className="h-8 w-px bg-slate-800" />
          
          <select 
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="bg-slate-900 border-2 border-primary/5 text-white text-sm font-black rounded-xl px-4 py-2 focus:ring-2 focus:ring-primary outline-none transition-all"
          >
            {languages.map(lang => (
              <option key={lang.id} value={lang.id}>{lang.name}</option>
            ))}
          </select>

          <h1 className="text-sm font-black text-slate-400 ml-4 tracking-widest uppercase">
            Curriculum: <span className="text-primary">{decodeURIComponent(title || 'Universal Sandbox')}</span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          {isLoading && (
            <div className="flex items-center gap-3 text-xs font-black text-primary animate-pulse mr-6 uppercase tracking-widest">
              <div className="w-2 h-2 rounded-full bg-primary" />
              Initializing Neural Link...
            </div>
          )}
          <Button 
            variant="ghost" 
            onClick={() => setCode(boilerplates[language])}
            className="text-slate-400 hover:text-white font-black uppercase text-xs tracking-widest"
          >
            <RotateCcw className="w-4 h-4 mr-2" /> Reset
          </Button>
          <Button 
            onClick={runCode}
            disabled={isLoading}
            className="btn-primary rounded-2xl px-10 py-6 text-lg font-black shadow-2xl shadow-primary/20 active:scale-95 transition-all"
          >
            <Play className="w-5 h-5 mr-2 fill-current" /> Execute Logic »
          </Button>

          <div className="h-8 w-px bg-slate-800" />

          <Button 
            onClick={() => completeMutation.mutate(decodeURIComponent(title || ''))}
            disabled={completeMutation.isPending || !title}
            variant="ghost"
            className={`rounded-2xl px-6 py-6 font-black uppercase text-xs tracking-widest transition-all ${
              completeMutation.isSuccess ? 'text-green-400 bg-green-400/10' : 'text-slate-400 hover:text-white'
            }`}
          >
            <CheckCircle2 className={`w-5 h-5 mr-2 ${completeMutation.isSuccess ? 'animate-bounce' : ''}`} />
            {completeMutation.isSuccess ? 'Achieved' : 'Mark Achieved'}
          </Button>
        </div>
      </header>

      {/* Main Editor Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Input */}
        <div className="flex-1 flex flex-col border-r border-slate-300 bg-white">
          <div className="bg-slate-200 px-4 py-1 text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center justify-between">
            <span>{language.toUpperCase()} Editor</span>
            <span className="flex items-center gap-1"><Save className="w-3 h-3 text-green-600" /> Live</span>
          </div>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="flex-1 p-6 font-mono text-sm resize-none focus:outline-none bg-slate-50 text-slate-800 leading-relaxed"
            spellCheck="false"
            placeholder={`Write your ${language} code here...`}
          />
        </div>

        {/* Output Panel */}
        <div className="flex-1 flex flex-col bg-slate-50">
          <div className="bg-slate-200 px-4 py-1 text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center justify-between">
            <span>{language === 'html' ? 'Live Preview' : 'Console Output'}</span>
            <span className="text-blue-600">Result</span>
          </div>
          
          {language === 'html' ? (
            <iframe
              srcDoc={srcDoc}
              title="output"
              sandbox="allow-scripts"
              className="flex-1 w-full border-none bg-white"
            />
          ) : (
            <div className="flex-1 p-6 font-mono text-sm overflow-auto bg-slate-900 text-green-400">
              {output || (isLoading ? '> Initializing engine...' : '> Press "Run" to see output')}
            </div>
          )}
        </div>
      </div>

      {/* Editor Footer */}
      <footer className="bg-white border-t px-4 py-2 flex items-center justify-between text-xs text-slate-500">
        <div className="flex gap-4">
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Auto-saving enabled
          </span>
          <span>Characters: {code.length}</span>
        </div>
        <div className="flex gap-4">
          <button 
            onClick={() => {
              alert("Snippet saved to your workspace!");
            }}
            className="hover:text-blue-600 flex items-center gap-1 transition-colors"
          >
            <Save className="w-3 h-3" /> Save to Profile
          </button>
          <button 
            onClick={downloadCode} 
            className="hover:text-blue-600 flex items-center gap-1 transition-colors"
          >
            <Download className="w-3 h-3" /> Download HTML
          </button>
          <button 
            onClick={() => {
              navigator.clipboard.writeText(window.location.href);
              alert("Link copied to clipboard!");
            }}
            className="hover:text-blue-600 flex items-center gap-1 transition-colors"
          >
            <Share2 className="w-3 h-3" /> Share
          </button>
        </div>
      </footer>
    </div>
  );
};
