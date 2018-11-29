
package com.demo.asyncproc;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DemoAsyncProc {

	//Runtime.getRuntime().availableProcessors(); //This returns count of CPUs!
	public static void main(String[] args) {
		List<MyTask> tasks = IntStream
								.range(0, 10)
								.mapToObj(i -> new MyTask(1))
								.collect(Collectors.toList());
		System.out.println();
		System.out.println();
			
		runSequentially(tasks);
		System.out.println();
		System.out.println();
		
		useParallelStream(tasks);
		System.out.println();
		System.out.println();
		
		
		useCompletableFuture(tasks);
		System.out.println();
		System.out.println();
		
		
		useCompletableFutureWithExecutor(tasks);
		
	}

	// Sequential
	public static void runSequentially(List<MyTask> tasks) {
		System.out.println("============ Sequential Beg =============");
		long start = System.nanoTime();
		List<Integer> result = tasks
								.stream()
								.map(MyTask::calculate)
								.collect(Collectors.toList());
		
		long duration = (System.nanoTime() - start) / 1_000_000;
		
		System.out.printf("Processed %d tasks in %d millis\n", tasks.size(), duration);
		System.out.println(result);
		System.out.println("============ Sequential End =============");
	}

	// Parallel Stream
	public static void useParallelStream(List<MyTask> tasks) {
		System.out.println("============ Parallel Beg =============");
		long start = System.nanoTime();
		List<Integer> result = tasks
								.parallelStream()
								.map(MyTask::calculate)
								.collect(Collectors.toList());
		
		long duration = (System.nanoTime() - start) / 1_000_000;
		
		System.out.printf("Processed %d tasks in %d millis\n", tasks.size(), duration);
		System.out.println(result);
		System.out.println("============ Parallel End =============");
	}

	// CompletableFutures
	public static void useCompletableFuture(List<MyTask> tasks) {
		System.out.println("============ CompletableFutures Beg =============");
		
		long start = System.nanoTime();
		
		List<CompletableFuture<Integer>> futures 
				= tasks
					.stream()
					.map(t -> CompletableFuture.supplyAsync(() -> t.calculate()))
					.collect(Collectors.toList());

		List<Integer> result 
				= futures
					.stream()
					.map(CompletableFuture::join)
					.collect(Collectors.toList());
		
		long duration = (System.nanoTime() - start) / 1_000_000;
		
		System.out.printf("Processed %d tasks in %d millis\n", tasks.size(), duration);
		System.out.println(result);
		System.out.println("============ CompletableFutures End =============");
	}

	// CompletableFutures with Custom Executor
	public static void useCompletableFutureWithExecutor(List<MyTask> tasks) {
		System.out.println("============ CompletableFutures with Custom Executor Beg =============");

		long start = System.nanoTime();
		
		ExecutorService executor = Executors.newFixedThreadPool(Math.min(tasks.size(), 10));
		
		List<CompletableFuture<Integer>> futures 
				= tasks
					.stream()
					.map(t -> CompletableFuture.supplyAsync(() -> t.calculate(), executor))
					.collect(Collectors.toList());

		List<Integer> result 
				= futures
					.stream()
					.map(CompletableFuture::join)
					.collect(Collectors.toList());
		
		long duration = (System.nanoTime() - start) / 1_000_000;
		
		System.out.printf("Processed %d tasks in %d millis\n", tasks.size(), duration);
		System.out.println(result);
		
		executor.shutdown();
		
		System.out.println("============ CompletableFutures with Custom Executor End =============");
	}

}
