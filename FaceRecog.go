package main

import (
	"fmt"
	"log"
	"os/exec"
	"os"
)

func main() {
	cmd := Parse()
	fmt.Println("Starting")
	err := cmd.Run()
	if err != nil {
		log.Println(err)
	}
	cmd.Wait()
}

func Parse()(command exec.Cmd){
	cmd := exec.Command("python","recognition.py",
		"--long_distance",os.Getenv("LONG_DISTANCE"))
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return *cmd
	// fmt.Println(output)
}

func ErrorHandle(err error){
	if err != nil{
		fmt.Println(err)
	}
} 
