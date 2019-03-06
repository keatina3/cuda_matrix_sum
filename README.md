<!DOCTYPE html>
<html lang="en" dir="ltr" prefix="content: http://purl.org/rss/1.0/modules/content/  dc: http://purl.org/dc/terms/  foaf: http://xmlns.com/foaf/0.1/  og: http://ogp.me/ns#  rdfs: http://www.w3.org/2000/01/rdf-schema#  schema: http://schema.org/  sioc: http://rdfs.org/sioc/ns#  sioct: http://rdfs.org/sioc/types#  skos: http://www.w3.org/2004/02/skos/core#  xsd: http://www.w3.org/2001/XMLSchema# ">
  <head>
    <meta charset="utf-8" />
<script>var _paq = _paq || [];(function(){var u=(("https:" == document.location.protocol) ? "https://www.tchpc.tcd.ie/piwik/" : "http://www.tchpc.tcd.ie/piwik/");_paq.push(["setSiteId", "13"]);_paq.push(["setTrackerUrl", u+"piwik.php"]);_paq.push(["setDoNotTrack", 1]);_paq.push(["trackPageView"]);_paq.push(["setIgnoreClasses", ["no-tracking","colorbox"]]);_paq.push(["enableLinkTracking"]);var d=document,g=d.createElement("script"),s=d.getElementsByTagName("script")[0];g.type="text/javascript";g.defer=true;g.async=true;g.src=u+"piwik.js";s.parentNode.insertBefore(g,s);})();</script>
<meta name="Generator" content="Drupal 8 (https://www.drupal.org)" />
<meta name="MobileOptimized" content="width" />
<meta name="HandheldFriendly" content="true" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<link rel="shortcut icon" href="/core/misc/favicon.ico" type="image/vnd.microsoft.icon" />
<link rel="canonical" href="/node/1677" />
<link rel="shortlink" href="/node/1677" />
<link rel="delete-form" href="/node/1677/delete" />
<link rel="delete-multiple-form" href="/admin/content/node/delete?node=1677" />
<link rel="edit-form" href="/node/1677/edit" />
<link rel="version-history" href="/node/1677/revisions" />
<link rel="revision" href="/node/1677" />

    <title>5615 - 01 - Matrix add up and reduce - parallel calculation | MSc in HPC</title>
    <link rel="stylesheet" href="/sites/default/files/css/css__J544U28r_xLn9W-jXCOajA7p3m0jmynvSw9ecXjW7Q.css?pnbvfs" media="all" />
<link rel="stylesheet" href="/sites/default/files/css/css_rkGCu7tF-F5eqr4Rt2i1iTwnYSYBglhVrH6Srf2jbvw.css?pnbvfs" media="all" />
<link rel="stylesheet" href="/sites/default/files/css/css_Z5jMg7P_bjcW9iUzujI7oaechMyxQTUqZhHJ_aYSq04.css?pnbvfs" media="print" />

    
<!--[if lte IE 8]>
<script src="/sites/default/files/js/js_VtafjXmRvoUgAzqzYTA3Wrjkx9wcWhjP0G4ZnnqRamA.js"></script>
<![endif]-->

  </head>

<article data-history-node-id="1677" role="article" about="/node/1677" class="node node--type-assignment node--view-mode-full clearfix">
  <header>
    
        
          <div class="node__meta">
        <article typeof="schema:Person" about="/user/18" class="profile">
  <a href="/blog/18">View recent blog entries</a></article>

        <span>
          Submitted by <span class="field field--name-uid field--type-entity-reference field--label-hidden"><a title="View user profile." href="/user/18" lang="" about="/user/18" typeof="schema:Person" property="schema:name" datatype="" class="username">jose</a></span>
 on <span class="field field--name-created field--type-created field--label-hidden">Tue, 2019/02/12 - 10:21</span>
        </span>
        
      </div>
      </header>
  <div class="node__content clearfix">
    
  <div class="field field--name-field-course-reference field--type-entity-reference field--label-inline">
    <div class="field__label">Course Reference</div>
              <div class="field__item"><a href="/node/5" hreflang="en">5615</a></div>
          </div>

  <div class="field field--name-field-assignment-number field--type-integer field--label-inline">
    <div class="field__label">Assignment Number</div>
              <div class="field__item">1</div>
          </div>

  <div class="field field--name-field-due-date field--type-datetime field--label-inline">
    <div class="field__label">Due Date</div>
              <div class="field__item"><time datetime="2019-03-01T17:00:00Z" class="datetime">Fri, 2019/03/01 - 17:00</time>
</div>
</div>

<div class="clearfix text-formatted field field--name-body field--type-text-with-summary field--label-hidden field__item"><h2>Matrix add up and reduce - parallel calculation</h2>

<p>The goal of this assignment is to reduce a matrix to a single value, adding its values up, by writing a basic CUDA implemention. ===================================================================================================================</p>

<p><strong>Task 1 - CPU calculation</strong> <em>(Only in the CPU)</em></p>

<p>Write C or C++ code that allocates a <strong>floating</strong> point matrix (not necessarily square!) of size <em>n</em> x <em>m</em> (both with a default value of 10 and that can be specified independently by passing command line arguments, <em>-n</em> and <em>-m</em>), in which each element is initialized by calling <em>((float)(drand48())*2.0)-1.0</em> (so the values will be between -1 and 1).</p>

<p>Use a fixed seed by seeding srand48 with "123456". Include also a command line option (<em>-r</em>) to seed the number of milliseconds instead. Implement the following:</p>

<ul><li>A function that adds together the absolute value of each element of each row (into a vector of size <em>n</em>)</li>
	<li>A function that adds together the absolute value of each element of each column (into a vector of size <em>m</em>)</li>
	<li>A reduce function that reduces either vector to a single value, by adding up the components.</li>
</ul><p>Add code to track the amount of time that each one of the functions take, and add a command line argument to display the timing (-t). Give a short explanation about the difference in performace for each one of them.</p>

<p><strong>Task 2 - parallel implementation</strong> <em>(In the CPU and GPU)</em></p>

<p>To the source code from the previous task, add cuda code (feel free to use the <em>makefileCpp, makefileExternC, cudaPrint, thread1d, thread2d, thread2dStl, matrixMultiplication</em> and <em>syncThreadsTest</em> examples that have been provided) to implement your own cuda versions of the calculation (so one way of adding rows, one way of adding columns, and a reduce operation). Use only the global memory and the cuda core registers, with the matrix having the same values as in the CPU case. You can organize the grid in whatever way you want (in both size and dimension). You will need to copy back the results to the RAM; compare them with the values obtained in the CPU. Add code to track the amount of time that the calculation of each one of the norms takes, and add a command line argument to display both the CPU and GPU timings for each norm next to each other.</p>

<p><strong>Task 3 - performance improvement</strong></p>

<p>Test the code for square matrices with row and column sizes <em>n</em>=<em>m</em>=1000,5000,10000 and 30000 with a number of threads per block equal to 4, 8, 16, 32, 64, 128, 256, 512 and 1024 (remember that you will have to factor that number if you are using a multidimensional grid). Calculate the speedups and precisions (errors) compared to the CPU versions.</p>

<p>Look at which operation perform better or worse using different kinds of grids, and, considering the obtained speedup, explain which ones are worth computing this way in Cuda and which ones are not.</p>

<p><strong>Task 4 - double precision testing</strong></p>

<p>Copy and paste your code into a different folder and replace the floats for doubles (or write both implementations in the same code, if you prefer) - then calculate the speedups and precisions for the optimal number of threads per block (you can use the same optimal number as for the single precision case) and row and column sizes 1000,5000,10000 and 30000. Compare the results obtained with the obtained single precision results.</p>

<p>===================================================================================================================</p>

<p>Submit a tar ball with your source code files (including a working Makefile), speedup graphs and a writeup of what you did and any observations you have made on the behaviour and performance of your code.</p>

<p><strong>Note:</strong> Do not forget to set up your PATH and LD_LIBRARY_PATH variables in your .bashrc, otherwise nvcc won't work! You have the instructions in the slide 65 of the pdf of the course.</p>

<p><strong>Note:</strong> Do not use atomic functions, libraries (other than cuda and C++'s stl) or other kinds of memory other than the global and per-thread registers (yet - plenty of time for that later on!).</p>

<p><strong>Note:</strong> Marks will be deducted for tarbombing. http://en.wikipedia.org/wiki/Tar_%28computing%29#Tarbomb</p>

<p><strong>Note:</strong> Extra marks will be given for separating the C/C++ and cuda code and providing a Makefile that works. You can find examples on how to do that on the "makefileCpp" and "makefileExternC" sample code.</p>

<p><strong>Note:</strong> Remember that the code must work for non-square matrices too (and with matrices that have negative elements), even when, for this assignment, we are using square matrices with positive numbers for benchmarking. You can run use <em>cuda-memcheck</em> to test that, as in: cuda-memcheck ./my_exec</p>

<p><strong>Note:</strong> When you are benchmarking the performance of your code, you can check the current load on cuda01 with the <em>nvidia-smi</em> command.</p>

<p>===================================================================================================================</p></div>
      
  </div>
</article>
