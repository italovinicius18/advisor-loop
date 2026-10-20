import subprocess
P = "/home/italo/advisor-loop/kubernetes/campaign-pods"
NAMES = ["sf100-rep-replay", "sf100-rep-replay-r2", "sf100-rep-replay-r3", "sf100-rep-a0-heuristic", "sf100-rep-a2-haiku", "sf500-rep-replay", "sf500-rep-replay-r2", "sf500-rep-replay-r3", "sf500-rep-a0-heuristic", "sf500-rep-a2-haiku"]
SSH = ["ssh", "-i", "/home/italo/.ssh/rnp", "-p", "2200", "italo-unb@200.159.254.113"]
SCP = ["scp", "-i", "/home/italo/.ssh/rnp", "-P", "2200"]
files = [f"{P}/advisor-controller-{n}.yaml" for n in NAMES] + [f"{P}/queue_rep.sh", f"{P}/advisor-queue-pod.yaml"]
subprocess.run(SSH + ["rm -rf ~/queue_rep && mkdir -p ~/queue_rep"], check=True)
subprocess.run(SCP + files + ["italo-unb@200.159.254.113:queue_rep/"], check=True)
remote = " && ".join([
    "cd ~/queue_rep",
    "cp queue_rep.sh queue_incluster.sh",
    "kubectl delete pod advisor-queue -n italo-unb --ignore-not-found --wait=true",
    "kubectl delete cm advisor-queue -n italo-unb --ignore-not-found",
    "kubectl create cm advisor-queue -n italo-unb --from-file=queue_incluster.sh "
    + " ".join(f"--from-file=advisor-controller-{n}.yaml" for n in NAMES),
    "kubectl apply -f advisor-queue-pod.yaml",
    "sleep 20",
    "kubectl get pods -n italo-unb --no-headers | grep -E 'advisor-queue|rep-'",
    "kubectl logs advisor-queue -n italo-unb",
])
p = subprocess.run(SSH + [remote], capture_output=True, text=True)
print(p.returncode, p.stdout, p.stderr[-800:])
