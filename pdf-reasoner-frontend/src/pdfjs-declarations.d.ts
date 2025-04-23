declare module 'pdfjs-dist/build/pdf' {
    const pdfjsLib: typeof import('pdfjs-dist/types/src/pdf');
    export = pdfjsLib;
  }